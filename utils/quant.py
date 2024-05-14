import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

index = 0


@torch.no_grad()
def _high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float("nan")))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all,
        )
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all,
        )

        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask

    return sum_order, scale_tensor_all


def gptq_quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def high_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


# NOTE: Normal quantizer utilized in GPTQ
class GPTQQuantizer(nn.Module):

    def __init__(self, shape=1):
        super(GPTQQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = gptq_quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return gptq_quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print("CUDA extension not installed.")


# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module):

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
        self.register_buffer("scales", torch.zeros((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.register_buffer(
            "qweight", torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(
            torch.int
        )
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(
                    x, self.qweight, y, self.scales, self.zeros
                )
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError("Only supports a single token currently.")


def make_quant3(module, names, name="", faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            setattr(
                module,
                attr,
                Quant3Linear(tmp.in_features, tmp.out_features, faster=faster),
            )
    for name1, child in module.named_children():
        make_quant3(
            child, names, name + "." + name1 if name != "" else name1, faster=faster
        )


# NOTE: HightQuantizer from PB-LLM
class HighQuantizer(nn.Module):

    def __init__(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        grouprows=1,
        shape=1,
    ):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.grouprows = grouprows

    def calibrate(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1:
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = high_quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            if self.grouprows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x, blocki=None):
        if self.ready():
            return high_quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class LowQuantizer(nn.Module):

    def __init__(self, weight, method="xnor", groupsize=-1):
        super().__init__()
        oc, ic = weight.shape
        if groupsize == -1:
            groupsize = ic
        self.groupsize = groupsize
        self.n_groups = math.ceil(ic / groupsize)
        if "bit" in method:
            self.register_buffer("maxq", torch.tensor(1))
            self.register_buffer("zero", torch.zeros(self.n_groups, oc, 1))

        self.register_buffer("scale", torch.zeros(self.n_groups, oc, 1))
        self.register_buffer("mean", torch.zeros(self.n_groups, oc, 1))
        self.method = method

    def calibrate(self, w, mask=None, groupi=0, order=2):
        if self.method == "xnor":
            # TODO: seems to have problem
            w_mean = w.mean(-1).view(-1, 1)  # oc, ic(blocksize)
            self.mean[groupi] = w_mean
            w = w - w_mean  # oc, ic(blocksize)
            # non_zero_nums = (w != 0).float().sum(-1,keepdim=True)
            # scale = w.abs().sum(-1,keepdim=True)/(non_zero_nums+1e-5)
            scale = w.abs().mean(-1, keepdim=True)
            # TODO: search mean and scale
        elif self.method == "sign":
            # w_relu=F.relu(w)
            # scale=w_relu.sum()/((w>0).float().sum()+1e-5)
            scale = F.relu(w).mean(-1, keepdim=True)
            # scale=w.abs().mean(-1,keepdim=True)
            # scale=w.mean(-1,keepdim=True)
        elif self.method == "braq":  # The method used in paper
            w, scale = _high_order_residual(w, mask, order=order)
            scale = scale.view(-1, 1)
        elif self.method == "rtn":
            scale = w.abs().mean(-1, keepdim=True) + 1e-5
        elif self.method in ["no", "prune"]:
            return
        elif self.method in ["2bit", "4bit"]:
            w = w
            dev = w.device
            if self.method == "2bit":
                self.maxq.fill_(3)
            elif self.method == "4bit":
                self.maxq.fill_(7)
            self.maxq = self.maxq.to(dev)
            self.scale = self.scale.to(dev)
            self.zero = self.zero.to(dev)
            w = w.flatten(1)
            tmp = torch.zeros(w.shape[0], device=dev)
            xmin = torch.minimum(w.min(1)[0], tmp)
            xmax = torch.maximum(w.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1

            scale = (xmax - xmin) / self.maxq
            scale = scale.reshape(-1, 1)
            self.zero[groupi] = torch.round(-xmin / scale[groupi]).reshape(-1, 1)
        else:
            raise NotImplementedError(f"method {self.method} not implemented")
        self.scale[groupi] = scale
        self.scale.to(w.device)

    def quantize(self, w, groupi=0):
        if w.device != self.scale.device:
            self.scale = self.scale.to(w.device)
            self.mean = self.mean.to(w.device)

        if self.method == "xnor":
            # return torch.zeros_like(w)
            w_mean = self.mean[groupi]
            w = w - w_mean  # oc, ic
            w = w.sign()
            # TODO remove to
            w = w * self.scale[groupi]
            w += w_mean

        elif self.method == "sign":
            w = (w > 0).float()
            w *= self.scale[groupi]
        elif self.method == "rtn":
            w = F.relu(w)
            w_int = (w / self.scale[groupi]).round().clamp(0, 1)
            w = w_int * self.scale[groupi]
        elif self.method in ["2bit", "4bit"]:
            q = torch.clamp(
                torch.round(w / self.scale[groupi]) + self.zero[groupi], 0, self.maxq
            )
            w = self.scale[groupi] * (q - self.zero[groupi])
        elif self.method == "prune":
            return torch.zeros_like(w)
        return w
