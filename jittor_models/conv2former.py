import jittor as jt
import jittor.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.Identity()

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x

class ConvMod(nn.Module):
    def __init__(self, kernel_size, dim, dilation=1):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Identity(),
                nn.Conv2d(dim, 
                          dim, 
                          kernel_size, 
                          padding=(kernel_size + (kernel_size - 1) * (dilation - 1))//2, 
                          groups=dim, 
                          dilation=dilation)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)   
        a = self.att(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, kernel_size, dim, mlp_ratio=4, drop_path=0.15, dilation=1):
        super().__init__()
        
        self.attn = ConvMod(kernel_size, dim, dilation=dilation)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1 = layer_scale_init_value * jt.ones((dim))
        self.layer_scale_1.requires_grad = True
        self.layer_scale_2 = layer_scale_init_value * jt.ones((dim))
        self.layer_scale_2.requires_grad = True
        
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def execute(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x
      
class LayerNorm(nn.Module):
    r""" From conv2former (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = jt.ones(normalized_shape)
        self.weight.requires_grad = True
        self.bias = jt.zeros(normalized_shape)
        self.bias.requires_grad = True
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def execute(self, x):
        if self.data_format == "channels_last":
            return nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / jt.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv2Former(nn.Module):
    def __init__(self, kernel_size, img_size=224, dilation=1, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 mlp_ratios=[4, 4, 4, 4], num_heads=[2, 4, 10, 16], layer_scale_init_value=1e-6, head_init_scale=1., 
                 drop_path_rate=0., drop_rate=0., **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Identity(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Identity(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=2, stride=2, bias=False),
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=stride, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], dilation=dilation, kernel_size=kernel_size, drop_path=dp_rates[cur + j], mlp_ratio=mlp_ratios[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Sequential(
                nn.Conv2d(dims[-1], dims[-1], 1),
                nn.Identity(),
                LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        )
        self.pred = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            jt.init.trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def execute_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        return x # global average pooling, (N, C, H, W) -> (N, C)

    def execute(self, x):
        x = self.execute_features(x)
        x = self.head(x)
        x = self.pred(x.mean([-2, -1]))
        return x
    
def conv2former_n(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[64, 128, 256, 512], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 8, 2], **kwargs)
    return model

def conv2former_t(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[72, 144, 288, 576], mlp_ratios=[4, 4, 4, 4], depths=[3, 3, 12, 3], **kwargs)
    return model

def conv2former_s(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[72, 144, 288, 576], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 32, 4], **kwargs)
    return model

def conv2former_b(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[96, 192, 384, 768], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    return model

def conv2former_l(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[128, 256, 512, 1024], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 48, 4], **kwargs)
    return model


if __name__ == "__main__":
    inputs = jt.randn(1,3,224,224)
    model = conv2former_t()
    model(inputs)