import time 
import numpy as np 
import torch 
import torch.nn as nn 
from utils.sparsegpt import SparseGPT 
from utils.layerwrapper import WrappedGPT
from datautils import get_loaders 
from utils.quant import Quantizer
from scipy.optimize import linear_sum_assignment
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

from utils.ablate import AblateGPT
# from autozc.structures.tree_engine import GPTree

def lexsort(keys, dim=-1):
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    return idx

def maximize_total_value(matrix):
    # linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(matrix, maximize=True) 
    return col_indices

def check_outlier_mean(mask, threshold, name=""):
    W = mask
    count = 0
    total_params = 0

    max_shred = torch.mean(W) * threshold
    count += (W > max_shred).sum().item()
    total_params += W.numel()

    # plot_outlier(W.cpu().numpy(), max_shred, f"outlier_distribution_{name}.png")

    outlier_ratio = float(count) / total_params * 100

    return outlier_ratio

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:     
        layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if hasattr(model, 'hf_device_map') and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device) # ori: 128
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 

    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def prune_wanda(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # llama     
    torch.cuda.empty_cache()

    model = model.to(device)
    layers = model.model.layers
    for i in range(len(layers)):
        if i == 0 or i == len(layers)-1:
            continue 

        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        else:
            dev = device
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
    
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        try:
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        except RuntimeError:
            breakpoint()
                
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, dataloader, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        # s1: 81.97
        # if i == 0 or i == len(layers)-1:
        #     continue
            
        # s2: 25% 
        # ratio=0.25
        # if i < int(ratio * len(layers)) and i > int((1-ratio)*len(layers)):
        #     continue

        # s3: first 3 and last 3
        # if i < 3 or i > len(layers)-4:
        #     continue

        # s4: within layer 
        # see below
            
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            # if "attn" in name:
            #     continue # filter out the mlp layer 
            
            # s4
            # if i == 0 and 'mlp' in name:
            #     continue
            # if i == len(layers)-1 and 'mlp' in name:
            #     continue

            # s5: first 3 and last 3 
            # if i < 3 and 'mlp' in name:
            #     continue
            # if i > len(layers)-4 and 'mlp' in name:
            #     continue 

            # s6: first 4 and last 4 
            if i < 4 and 'mlp' in name:
                continue
            if i > len(layers)-5 and 'mlp' in name:
                continue 
            
            # s7: first 5 and last 5 
            # if i < 5 and 'mlp' in name:
            #     continue
            # if i > len(layers)-6 and 'mlp' in name:
            #     continue

            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_gblm(args,
               model,
               dataloader,
               device=torch.device('cuda:0'),
               prune_n=0,
               prune_m=0,
               layer_no=-1):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(
            args.gradient_path, map_location=torch.device('cpu'))

    # print('loading calibdation data')
    # dataloader, _ = get_loaders(
    #     'wikitext2',
    #     nsamples=args.nsamples,
    #     seed=args.seed,
    #     seqlen=2048,
    #     tokenizer=tokenizer)
    print('dataset loading complete')
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f'model.layers.{i}' in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f'model.layers.{i}']
            inps, outs, position_ids = inps.to(dev), outs.to(
                dev), position_ids.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(
                subset[name], layer_id=i, layer_name=name)

        def add_batch(name):

            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(
                add_batch(name)))  ## this is a important function.
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for sub_i, name in enumerate(subset):
            indexed_name = f'{name}_layer_{i}'
            print(f'pruning layer {i} name {name}')
            tmp_weight = subset[name].weight.data.detach().clone()
            W_metric = torch.abs(tmp_weight) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))
            # W = |W|
            if not args.gradient_inv:
                # small_value = torch.tensor(1e-8, dtype=gradients[indexed_name].dtype, device=gradients[indexed_name].device)
                W_metric_grad = torch.abs(tmp_weight) * torch.abs(
                    gradients[indexed_name].to(device=W_metric.device))
                W_metric = W_metric.to(dtype=torch.float32) + W_metric_grad.to(
                    dtype=torch.float32)  #+ small_value)
            else:
                small_value = torch.tensor(
                    1e-8,
                    dtype=gradients[indexed_name].dtype,
                    device=gradients[indexed_name].device)
                gradient_inv = 1 / (
                    torch.abs(gradients[indexed_name]) + small_value)
                W_metric = W_metric.to(dtype=torch.float32) * gradient_inv.to(
                    device=W_metric.device).to(dtype=torch.float32)

            W_mask = (torch.zeros_like(W_metric) == 1
                      )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(
                            1, ii +
                            torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) >
                           0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f'alpha found {alpha} sparsity {cur_sparsity:.6f}')
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] *
                                                  args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



def prune_ri(args, model, dataloader, device=torch.device('cuda:0'), prune_n=0, prune_m=0, layer_no=-1):
    """plug and play based on magnitude pruning"""
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_abs = torch.abs(W)
            sum_abs_cols = torch.sum(W_abs, dim=0, keepdim=True)
            sum_abs_rows = torch.sum(W_abs, dim=1, keepdim=True)
            R = W_abs / (sum_abs_cols + sum_abs_rows - W_abs)  # Subtract W_abs to avoid counting the weight itself twice

            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(R.shape[1]):
                    if ii % prune_m == 0:
                        tmp = R[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(
                            1, ii +
                            torch.topk(tmp, prune_n, dim=1, largest=True)[1],  # Change largest to True to prune least relevant weights
                            True)
            else:
                thresh = torch.sort(R.flatten())[0][int(
                    R.numel() * args.sparsity_ratio)].cpu()
                W_mask = (R <= thresh)  # Prune weights with relevance index less than or equal to the threshold

            W[W_mask] = 0

def fun1_standardize(M):
    # M can be weight or activation or gradient
    M_mean = M.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
    M = M - M_mean
    std = M.view(M.size(0), -1).std(dim=1).view(-1, 1) + 1e-5
    M = M / std.expand_as(M)
    return M

def prune_ria(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.dataset,nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen, model=args.model)
    print("dataset loading complete")
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                # inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            if args.gptq:
                print('Quantizing ...')
                wrapped_layers[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
            
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            if args.prune_method == "wanda":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "ria":
                W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.a
                W_metric = fun1_standardize(W_metric) # NOTE THAT HERE WE CHANGE SOMETHING
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                # Directly N:M
                W_mask = (torch.zeros_like(W_metric) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                
                if args.semi_sparse_acc:
                    subset[name].weight = torch.nn.Parameter(to_sparse_semi_structured(((W_mask==0)*W)).half(), requires_grad=False)
                    subset[name].mask = W_mask==0
                else:
                    subset[name].weight.data[W_mask] = 0
            else:
                if args.per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                    
                if args.reconstruction:
                    wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_ria_outlier_structure_special(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    ##### calucalte outlier ratio
    all_layer_ratio = []  # step1: record the layer outlier ratio
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(args.dataset,nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen, model=args.model)
    
    print("dataset loading complete")
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()
        layer_wmetric = []  # step2: record the layer weight metric

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()
            # NOTE: WANDA
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            
            # NOTE: RIA
            W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.a
            
            # layer_wmetric.append(activation_data)

            layer_wmetric.append(W_metric)  # step2

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

        # step2
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        # step3: calculate the outlier ratio
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)  # step3
            print("layer outlier ratio", out_ratio, out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # step 4: adjust the outlier ratio by z-scaling
    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
        1
        / (all_layer_ratio.max() - all_layer_ratio.min())
        * args.Lamda  # 0.08 by default
    )
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio)

    all_layer_ratio = np.round(all_layer_ratio)

    # Here is the key step to adjust the outlier ratio with N:M
    for i in range(len(all_layer_ratio)):
        if all_layer_ratio[i] == 1.0:
            all_layer_ratio[i] = 2.0

    all_layer_ratio = prune_n - all_layer_ratio

    print("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        prune_n = int(all_layer_ratio[i])
        print("Layer {} prune_n {} prune_m {}".format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data
            # NOTE: WANDA
            # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
            #     wrapped_layers[name].scaler_row.reshape((1, -1))
            # )
            
            # NOTE: RIA
            W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))**args.a
            
            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(
                        inps[j].unsqueeze(0), attention_mask=attention_mask
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



def prune_advanced_ria(args, model, dataloader, device=torch.device('cuda:0'), prune_n=0, prune_m=0, layer_no=-1, alpha=0.5):
    layers = model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    def weighted_sum_ratio(W):
        row_norms = torch.sqrt((W**2).sum(dim=1, keepdim=True))
        col_norms = torch.sqrt((W**2).sum(dim=0, keepdim=True))
        return W / row_norms + W / col_norms

    print('dataset loading complete')
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f'model.layers.{i}' in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f'model.layers.{i}']
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(
                dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(
                subset[name], layer_id=i, layer_name=name)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(
                add_batch(name)))  ## this is a important function.
            
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids)[0]

        for h in handles:
            h.remove()
            
        for name in subset:
            print(f'pruning layer {i} name {name}')
            # X_norm = torch.norm(wrapped_layers[name].scaler_row.reshape((1, -1)), p=2, dim=0)
            W = subset[name].weight.data
            # W_abs = torch.abs(W)
            # sum_abs_cols = torch.sum(W_abs, dim=0, keepdim=True)
            # sum_abs_rows = torch.sum(W_abs, dim=1, keepdim=True)
            # R = W_abs / sum_abs_cols + W_abs / sum_abs_rows
            W_abs = torch.abs(subset[name].weight.data)
            log_norm = torch.log1p(weighted_sum_ratio(W_abs))
            RIA = log_norm * torch.pow(wrapped_layers[name].scaler_row.reshape((1, -1)), 0.25)

            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(RIA.shape[1]):
                    if ii % prune_m == 0:
                        tmp = RIA[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(
                            1, ii +
                            torch.topk(tmp, prune_n, dim=1, largest=True)[1],  # Pruning the least relevant weights
                            True)
            else:
                thresh = torch.sort(RIA.flatten())[0][int(
                    RIA.numel() * args.sparsity_ratio)].cpu()
                W_mask = (RIA <= thresh)

            W[W_mask] = 0


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers 
    else: 
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "OPT" in model.__class__.__name__:
                cache['position_ids'] = None 
            else:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name], gradient_path=args.gradient_path)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_prunerzero_seq":
                indexed_name = f'{name}_layer_{i}'
                prune_mask = gpts[name].get_prunerzero_mask(args.sparsity_ratio, prune_n, prune_m, indexed_name)
            elif "iter" in args.prune_method:
                prune_mask = None 

            indexed_name = f'{name}_layer_{i}'
            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, indexed_name=indexed_name)
            gpts[name].free()

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    
def prune_pruner_zero(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0, engine=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    with open(args.gradient_path, 'rb') as file:
        gradients = torch.load(
            args.gradient_path, map_location=torch.device('cpu'))
        
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            indexed_name = f'{name}_layer_{i}'
            print(f"pruning layer {i} name {name}")
            # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W = torch.abs(subset[name].weight.data)
            # X = wrapped_layers[name].scaler_row.reshape((1,-1))
            G = gradients[indexed_name]
            
            # W_metric = torch.abs(W) * torch.sqrt(X)
            W_metric = engine.forward(
                W.to(dtype=torch.float32),
                G.to(device=W.device, dtype=torch.float32),
            )
            assert W_metric is not None

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
