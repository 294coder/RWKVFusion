import torch
import torch.nn as nn
from einops import rearrange
import numbers
from model.module.helper_func import to_3d, to_4d


def normalization(norm_type, channels, spatial_size=None):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if norm_type == "gn":
        return nn.GroupNorm(32, channels)
    elif norm_type == "bn":
        return nn.BatchNorm2d(channels)
    elif norm_type == "ln":
        # dim = [channels, spatial_size, spatial_size]
        return LayerNorm(channels, LayerNorm_type="BiasFree")
    elif norm_type == "in":
        return nn.InstanceNorm2d(channels)
    elif norm_type == "ln_no_bias":
        return LayerNorm(channels, LayerNorm_type="BiasFree")
    elif norm_type == "ln_bias":
        return LayerNorm(channels, LayerNorm_type="WithBias")
    else:
        return nn.Identity()


# =====================Restormer LayerNorm=====================


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.eps = eps

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(sigma + self.eps) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.eps = eps

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) * torch.rsqrt(sigma + self.eps) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias", channel_first=True, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.channel_first = channel_first

        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim, eps=eps)
        else:
            self.body = WithBias_LayerNorm(dim, eps=eps)

    def forward(self, x):
        if x.ndim == 4:
            if not self.channel_first:
                # [b, h, w, c] -> [b, c, h, w]
                x = x.permute(0, -1, 1, 2)
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        elif x.ndim == 3:
            if not self.channel_first:
                return self.body(x)
            else:
                # [b, c, l] -> [b, l, c]
                x = x.permute(0, -1, 1)
                return self.body(x).permute(0, -1, 1)
        else:
            raise NotImplementedError(f"LayerNorm not implemented for {x.ndim}D tensor")


# =====================Restormer LayerNorm=====================


# =====================NAFNet LayerNorm=====================
# used it NAFNet
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class NAFLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# =====================NAFNet LayerNorm=====================

# =====================LLAMA RMSNorm=====================


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        embd_dim: int,
        channel_first: bool = False,
        normed_axis: int = None,
        num_dims: int = None,
        eps: float = 1e-6,
        only_affine: bool = False,
    ):
        super().__init__()
        """
        Root Mean Square Layer Normalization for input tensor,
        shaped as [batch_size, sequence_length, hidden_size]
        """
        self.only_affine = only_affine
        if not only_affine:
            # assert num_dims is not None, "num_dims must be provided if only_affine is False"
            # _rep = [1 if i != normed_axis else embd_dim for i in range(num_dims)]
            # self.scale = nn.Parameter(torch.ones(_rep))
            self.scale = nn.Parameter(torch.ones(1))
        self.normed_axis = normed_axis
        self.embd_dim = embd_dim
        self.channel_first = channel_first
        self.num_dims = num_dims
        self.eps = eps

    def forward_norm(self, x: torch.Tensor):
        x_dtype = x.dtype
        rrms = torch.rsqrt(
            torch.mean(x**2, dim=self.normed_axis, keepdim=True) + self.eps
        )
        x = (x * rrms).to(dtype=x_dtype)
        if self.only_affine:
            return x
        else:
            return x * self.scale

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            if not self.channel_first:
                # [b, h, w, c] -> [b, c, h, w]
                x = x.permute(0, -1, 1, 2)
            h, w = x.shape[-2:]
            return to_4d(self.forward_norm(to_3d(x)), h, w)
        elif x.ndim == 3:
            if not self.channel_first:
                return self.forward_norm(x)
            else:
                # [b, c, h] -> [b, h, c]
                x = x.permute(0, -1, 1)
                return self.forward_norm(x).permute(0, -1, 1)
        else:
            raise NotImplementedError(f"LayerNorm not implemented for {x.ndim}D tensor")

    def __repr__(self):
        return f"RMSNorm(normed_axis={self.normed_axis}, embd_dim={self.embd_dim}, num_dims={self.num_dims}, only_affine={self.only_affine})"


if __name__ == "__main__":
    rms = RMSNorm(normed_axis=-1, embd_dim=256, num_dims=3)
    x = torch.randn(2, 256, 512)
    print(rms(x).shape)
