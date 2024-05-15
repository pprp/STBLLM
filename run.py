import time

import torch
import torch.nn as nn
from importlib.metadata import version

from utils.bigptq import BRAGPTQ
from utils.binary import Binarization
from utils.modelutils import find_layers
from utils.prune import (
    prune_wanda,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    find_layers,
    prune_ri,
    prune_ria,
    prune_gblm,
    prune_pruner_zero,
    prune_ria_outlier_structure_special,
)
from utils.layerwrapper import WrappedGPT
from utils.quant import GPTQQuantizer, LowQuantizer, HighQuantizer


# from autozc.structures.tree_engine import GPTree

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model or "Llama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map="auto"
        )
        model.seqlen = 2048

    return model


"""
The function is employed to calibrate and quantize models layer by layer.
"""


@torch.no_grad()
def quant_sequential_braqgptq(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            braq_quantizer = Binarization(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=args.groupsize,
            )
            gptq[name] = BRAGPTQ(
                subset[name],
                braq_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print(f"Quantizing the layer {name} ...")
            info = gptq[name].fasterquant(
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return model


@torch.no_grad()
def quant_sequential_pbllm(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue

            low_quantizer = LowQuantizer(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=args.groupsize,
            )
            
            # low_quantizer = Binarization(
            #     subset[name].weight,
            #     method=args.low_quant_method,
            #     groupsize=args.groupsize,
            # )
            
            high_quantizer = HighQuantizer(
                args.high_bit,
                perchannel=True,
                sym=False,
                mse=False,
            )
            wrapped_layers[name] = WrappedGPT(
                args,
                subset[name],
                layer_name=name,
                reconstruct=args.reconstruction,
                salient_metric=args.salient_metric,
                low_quantizer=low_quantizer,
                high_quantizer=high_quantizer,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in wrapped_layers:
            print(i, name)
            print("Quantizing ...")
            info = wrapped_layers[name].lowhightquant(
                args.low_frac, percdamp=args.percdamp, blocksize=args.groupsize
            )
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del wrapped_layers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return model



if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(",")))

    def list_of_floats(arg):
        return list(map(float, arg.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")

    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq", "ternary"],
        help="quantization method; `xnor` is the method using XNOR to adapt hardware calculation; `prune` is the method used in sparseGPTQ; braq is the method used in BiLLM",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )

    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "magnitude",
            "wanda",
            "sparsegpt",
            "ablate_mag_seq",
            "ablate_wanda_seq",
            "ablate_mag_iter",
            "ablate_wanda_iter",
            "search",
            "pruner-zero",
            "ablate_prunerzero_seq",
            "ablate_prunerzero_iter",
            "ri",
            "ria",
            "gblm",
            "ria_structure",
        ],
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--gradient_path",
        type=str,
        default="gradients/llama2/gradients_aggregrate_norm_l2_model_tinyllama-1.1b-480k-1t.pth",
        help="Path to the gradients",
    )
    # from ria
    parser.add_argument("--a", type=float, default=0.5, help="exponenet of activation")
    parser.add_argument(
        "--reconstruction",
        action="store_true",
        help="remaining weight reconstruction based on sparsegpt",
    )
    parser.add_argument(
        "--reallocation", action="store_true", help="Heuristic Channel Reallocation"
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--lsa", action="store_true", help="Linear Sum Assignment")
    parser.add_argument(
        "--semi_sparse_acc",
        action="store_true",
        help="using pytorch semi sparse acceleration. Only when sparsity type is 2:4",
    )
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument("--pbllm", action="store_true", help="use pbllm or not")
    parser.add_argument("--billm", action="store_true", help="use billm or not")

    parser.add_argument("--importance_score", type=str, default="sum", choices=["sum"])

    # hyperparameters for owl
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )

    parser.add_argument(
        "--Hyper_m",
        type=float,
        default=3,
    )

    parser.add_argument("--wbits", type=int, default=4, help="weight bits")

    parser.add_argument("--sym", action="store_true", help="symmetric quantization")

    parser.add_argument(
        "--high_bit",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--act_order",
        action="store_true",
        help="order of activation",
    )
    
    parser.add_argument(
        "--static_groups",
        action="store_true",
        help="static groups",
    )

    parser.add_argument("--low_frac", type=float, default=0.95, help="Target low_frac")

    args = parser.parse_args()
    assert args.groupsize == args.blocksize, "groupsize must be equal to blocksize"

    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{args.groupsize}_{args.salient_metric}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
    else:  # braq
        model = get_model(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model.eval()
        print(f"Available CUDA devices: {torch.cuda.device_count()}")

        if (
            "65b" in args.model or "70b" in args.model
        ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
            print("use device ", device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )

        # # prune after quant
        # start_time = time.time()
        # if args.sparsity_ratio != 0:
        #     print("pruning starts")
        #     if args.prune_method == "wanda":
        #         prune_wanda(
        #             args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif args.prune_method == "magnitude":
        #         prune_magnitude(
        #             args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif args.prune_method == "sparsegpt":
        #         prune_sparsegpt(
        #             args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif "ablate" in args.prune_method:
        #         prune_ablate(
        #             args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif "ria" == args.prune_method:
        #         prune_ria(
        #             args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif "ri" == args.prune_method:
        #         prune_ri(
        #             args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif "gblm" in args.prune_method:
        #         prune_gblm(
        #             args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     elif "ria_structure" in args.prune_method:
        #         prune_ria_outlier_structure_special(
        #             args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
        #         )
        #     # elif "pruner-zero" in args.prune_method:
        #     #     engine = GPTree.load_tree('./data/best_tree.json')
        #     #     prune_pruner_zero(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m, engine=engine)
        #     else:
        #         raise NotImplementedError
        # end_time = time.time()
        # print("pruning time: ", end_time - start_time)

        print("Begin quantizing ...")
        tick = time.time()
        model = quant_sequential_braqgptq(model, dataloader, device)
        print("quantization time:", time.time() - tick, "s")

    for dataset in ["wikitext2"]:
        # , "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model or "Llama" in args.model:
            from eval_ppl_utils import llama_eval

            llama_eval(model, testloader, device, dataset, args.log_wandb)

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
