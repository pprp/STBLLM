import time

import torch
import torch.nn as nn
from importlib.metadata import version

from bigptq import BRAGPTQ
from binary import Binarization
from modelutils import find_layers
from prune import prune_wanda, prune_magnitude, prune_sparsegpt, \
    prune_ablate, check_sparsity, find_layers, prune_ri, prune_ria, \
        prune_gblm, prune_pruner_zero, prune_advanced_ria
# from autozc.structures.tree_engine import GPTree


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

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
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        model.seqlen = 2048

    return model



if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune", "braq"],
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
        "--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", 
        "ablate_wanda_iter", "search", "pruner-zero", "ablate_prunerzero_seq", "ablate_prunerzero_iter",
        "ri", "ria", "gblm", 'advanced_ria']
    )
    parser.add_argument(
        "--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"]
    )
    parser.add_argument(
        '--sparsity_ratio', type=float, default=0, help='Sparsity level'
    )
    parser.add_argument(
        '--gradient_path', type=str, default="gradients/llama2/gradients_aggregrate_norm_l2_model_tinyllama-1.1b-480k-1t.pth",
        help='Path to the gradients'
    )

    args = parser.parse_args()
    groupsize = args.blocksize

    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"
    
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    
    
    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
    else: # braq
        model = get_model(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        model.eval()
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
        
        if "30b" in args.model or "65b" in args.model or \
            "70b" in args.model or "33b" in args.model or \
                "7b" in args.model: 
                # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
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
        
        # prune after quant
        start_time = time.time()
        if args.sparsity_ratio != 0:
            print("pruning starts")
            if args.prune_method == "wanda":
                prune_wanda(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            elif "ablate" in args.prune_method:
                prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif "ria" in args.prune_method:
                prune_ria(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            elif "ri" in args.prune_method:
                prune_ri(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            elif "gblm" in args.prune_method:
                prune_gblm(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            # elif "pruner-zero" in args.prune_method:
            #     engine = GPTree.load_tree('./data/best_tree.json')
            #     prune_pruner_zero(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m, engine=engine)
            elif "advanced_ria" in args.prune_method: 
                prune_advanced_ria(args, model, dataloader, device, prune_n=prune_n, prune_m=prune_m)
            else:
                raise NotImplementedError(f"Pruning method {args.prune_method} not implemented.")
        end_time = time.time()
        print("pruning time: ", end_time - start_time)
        
        

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)

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