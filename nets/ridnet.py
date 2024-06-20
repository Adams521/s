import torch.nn as nn
from nets import ops

def make_model(args, parent=False):
    return RIDNET(args)



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = ops.BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )    
    def forward(self, x):
        out = self.body(x)
        return out
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()
        # print("Block",in_channels,out_channels)
        self.r1 = ops.Merge_Run_dual(in_channels, in_channels)
        self.r2 = ops.ResidualBlock(in_channels, in_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        #self.g = ops.BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        # print("ridnet",x.shape)
        r1 = self.r1(x)            
        r2 = self.r2(r1)       
        r3 = self.r3(r2)
        #g = self.g(r3)
        out = self.ca(r3)

        return out
