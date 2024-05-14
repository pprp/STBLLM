import os
import math
import time

import torch
import torch.nn as nn
import transformers
from .quant import *
from utils.structure import structural_guassian_distribution

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

OUTPUTMASK = 1
DEBUG = False


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, args, layer, layer_id=0, 
                 layer_name="none",
                 reconstruct=True,
                 braq_quantizer=None,
                 salient_metric="hessian",
                 disable_gptq=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.reconstruct = reconstruct
        if self.reconstruct or args.gptq:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        
        if "up" in layer_name or "gate" in layer_name:
            self.out = torch.zeros((self.rows), device=self.dev)
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.sigmoid = nn.Sigmoid()
        
        self.braq_quantizer = braq_quantizer 
        self.salient_metric = salient_metric
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            out = out.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                out = out.reshape((-1, out.shape[-1]))
            inp = inp.t()
            out = out.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        if "up" in self.layer_name or "gate" in self.layer_name:
            self.out *= self.nsamples / (self.nsamples + tmp)

        if self.reconstruct:
            self.H *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp

        inp = inp.type(torch.float32)
        if "gate" in self.layer_name:
            out = (self.sigmoid(out) * out).type(torch.float32)
        elif "up" in self.layer_name:
            out = out.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        if "up" in self.layer_name or "gate" in self.layer_name:
            self.out += torch.mean(torch.abs(out), dim=1) / self.nsamples

        if self.reconstruct:
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

    # NOTE: SparseGPT
    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01, mask=None
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        try:
            H_tmp = H.clone()
            damp = percdamp * torch.mean(torch.diag(H_tmp))
            diag = torch.arange(self.columns, device=self.dev)
            H_tmp[diag, diag] += damp
            H_tmp = torch.linalg.cholesky(H_tmp)
            H_tmp = torch.cholesky_inverse(H_tmp)
            H_tmp = torch.linalg.cholesky(H_tmp, upper=True)
            Hinv = H_tmp

        except torch._C._LinAlgError:
            print("The matrix is not postive-definite, try a larger percdamp!")
            percdamp = 0.1
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = (
                        W1[:, i : (i + prune_m)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prune_m)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    # NOTE: RIA
    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        partition=3,
        orders=(1, 1, 2),
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    # NOTE: PB-LLM
    def lowhightquant(self, low_frac, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if not self.high_quantizer.ready():
            self.high_quantizer.calibrate(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        mask = None
        mask = torch.zeros_like(W, dtype=torch.bool)
        for groupi in range(self.low_quantizer.n_groups):
            st = groupi * self.low_quantizer.groupsize
            ed = min(st + self.low_quantizer.groupsize, self.columns)
            if self.salient_metric == "magnitude":
                saliency = torch.abs(W[:, st:ed])
                thresh = torch.sort(saliency.flatten())[0][
                    int(saliency.numel() * low_frac)
                ]
                mask[:, st:ed] = saliency <= thresh
            elif self.salient_metric == "hessian":
                tmp = (
                    W[:, st:ed] ** 2
                    / (torch.diag(H[st:ed, st:ed]).reshape((1, -1))) ** 2
                )
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * low_frac)]
                mask[:, st:ed] = tmp <= thresh
            else:
                raise NotImplementedError
            assert self.low_quantizer.groupsize % blocksize == 0
            self.low_quantizer.calibrate(
                W[:, st:ed] * mask[:, st:ed], mask[:, st:ed], groupi=groupi
            )
            # self.low_quantizer.calibrate(W[:,st:ed],mask[:,st:ed],groupi=groupi)

        if OUTPUTMASK:
            if os.path.exists("./outputs/mask") == False:
                os.mkdir("./outputs/mask")
            torch.save(
                mask,
                f"./outputs/mask/mask_{low_frac}_{self.layer.global_name.replace('/','_')}.pkl",
            )

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st
            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                q_high = self.high_quantizer.quantize(w)
                groupi = col_st // self.low_quantizer.groupsize
                q_low = self.low_quantizer.quantize(w, groupi)
                q = q_high * ~mask[:, col_st:col_ed] + q_low * mask[:, col_st:col_ed]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                if mask is not None:
                    mask1 = mask[:, col_st:col_ed]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    # TODO: use torch.kthvalue
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * low_frac)]
                    mask1 = tmp <= thresh

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q_high = self.high_quantizer.quantize(w.unsqueeze(1)).flatten()
                    # TODO: support groupsize>1
                    groupi = col_st // self.low_quantizer.groupsize
                    q_low = self.low_quantizer.quantize(
                        w.unsqueeze(1), groupi
                    ).flatten()
                    q = q_high * ~mask1[:, i] + q_low * mask1[:, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    # breakpoint()

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        return {"error": torch.sum(Losses).item()}

    # NOTE: BRAGPTQ
    def bragptqquant(
        self,
        blocksize=128,
        percdamp=0.01,
        partition=3,
        orders=(1, 1, 2),
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = (
                torch.zeros_like(W[:, st:ed], dtype=torch.bool)
                .unsqueeze(0)
                .repeat_interleave(partition, dim=0)
            )
            mask1, mask2, mask3 = structural_guassian_distribution(
                W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50
            )
            mask[0] = mask1
            mask[1] = mask2
            mask[2] = mask3

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                w = W[:, col_st:col_ed]

                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(
                        self.braq_quantizer.quantize(w, mask[i], order=orders[i])
                    )

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(
                        self.braq_quantizer.quantize(W1, mask[i], order=orders[i])
                    )

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
