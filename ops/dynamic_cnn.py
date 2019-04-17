import torch
from torch import nn
from torch.nn import functional as F
import fairseq


class Dynamic_Cnn_Wrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(Dynamic_Cnn_Wrapper, self).__init__()
        self.block = block
        self.dynamic_cnn = fairseq.modules.dynamic_convolution.DynamicConv1dTBC(input_size=block.bn3.num_features, kernel_size=5, padding_l=4, weight_dropout=0.1, weight_softmax=True, num_heads=4)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).permute(1, 0, 3, 4, 2)  # t, n, h, w, c
        x = x.view(self.n_segment, -1, c) # t, n * h * w, c
        x = self.dynamic_cnn(x)
        x = x.view(self.n_segment, -1, h, w, c).permute(1, 0, 4, 2, 3) # n, t, c, h, w
        x = x.view(nt, c, h, w)
        return x
    
def make_dynamic_cnn(net, n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet) or isinstance(net, archs.small_resnet.ResNet):
        net.layer2 = nn.Sequential(
            Dynamic_Cnn_Wrapper(net.layer2[0], n_segment),
            net.layer2[1],
            Dynamic_Cnn_Wrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            Dynamic_Cnn_Wrapper(net.layer3[0], n_segment),
            net.layer3[1],
            Dynamic_Cnn_Wrapper(net.layer3[2], n_segment),
            net.layer3[3],
            Dynamic_Cnn_Wrapper(net.layer3[4], n_segment),
            net.layer3[5],
        )
    else:
        raise NotImplementedError
        
        
class Light_Cnn_Wrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(Light_Cnn_Wrapper, self).__init__()
        self.block = block
        self.light_cnn = fairseq.modules.lightweight_convolution.LightweightConv1dTBC(input_size=block.bn3.num_features, kernel_size=5, padding_l=4, weight_dropout=0.1, weight_softmax=True, num_heads=4)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).permute(1, 0, 3, 4, 2)  # t, n, h, w, c
        x = x.contiguous().view(self.n_segment, -1, c) # t, n * h * w, c
        x = self.light_cnn(x)
        x = x.view(self.n_segment, -1, h, w, c).permute(1, 0, 4, 2, 3) # n, t, c, h, w
        x = x.contiguous().view(nt, c, h, w)
        return x
    
def make_light_cnn(net, n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet) or isinstance(net, archs.small_resnet.ResNet):
        net.layer2 = nn.Sequential(
            Light_Cnn_Wrapper(net.layer2[0], n_segment),
            net.layer2[1],
            Light_Cnn_Wrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            Light_Cnn_Wrapper(net.layer3[0], n_segment),
            net.layer3[1],
            Light_Cnn_Wrapper(net.layer3[2], n_segment),
            net.layer3[3],
            Light_Cnn_Wrapper(net.layer3[4], n_segment),
            net.layer3[5],
        )
    else:
        raise NotImplementedError