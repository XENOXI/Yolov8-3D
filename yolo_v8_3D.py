from cfg import *
from yololoss import make_anchors,dist2bbox
import torch
import torch.nn as nn
import cv2
from torchvision.ops import nms 
import math

class Conv2Plus1D(nn.Module):
    def __init__(self,in_ch,out_ch,k,s,p,g=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,(1,k,k),(1,s,s),(0,p,p),bias=False,dilation=1,groups=g),
            nn.Conv3d(out_ch,out_ch,(3,1,1),(1,1,1),(1,0,0),bias=False,dilation=1,groups=g)
        )

    def forward(self,x):
        return self.layers(x)

class Conv(nn.Module):
    def __init__(self,in_ch,out_ch,k,s=1,p=None,g=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if p is None:
            p = k//2   
        self.c = Conv2Plus1D(in_ch,out_ch,k,s,p,g)
        self.batchnorm = nn.BatchNorm3d(out_ch,eps=1E-3,momentum=0.03)
        self.silu = nn.SiLU(inplace=True)

            
    def forward(self,x):
        x = self.c(x)
        x = self.batchnorm(x)
        return self.silu(x)
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.batchnorm(self.c(x))

    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3,3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1,0)
        self.cv2 = Conv(c_ * 4, c2, 1, 1,0)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

            
    def forward(self, x):    
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Detect(nn.Module):
    def __init__(self,ch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = Conv(ch,ch,3,1,1)
        self.c2 = Conv(ch,ch,3,1,1)
        self.conv1 = nn.Conv3d(ch,4*reg_max,1,1,0)

        self.c3 = Conv(ch,ch,3,1,1)
        self.c4 = Conv(ch,ch,3,1,1)
        self.conv2 = nn.Conv3d(ch,num_of_cls,1,1,0)

    def forward(self,x):
        x1 = self.c1(x)
        x1 = self.c2(x1)
        x1 = self.conv1(x1)

        x2 = self.c3(x)
        x2 = self.c4(x2)
        x2 = self.conv2(x2)
        
        return torch.cat([x1,x2],1).transpose(1,2)
    
    def bias_init(self,stride):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        self.conv1.bias.data[:] = 1.0  # box
        self.conv2.bias.data[: num_of_cls] = math.log(5 / num_of_cls / (640 / stride) ** 2)  # cls (.01 objects, 80 classes, 640 img)
    

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
    
    


class YOLOv8_3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 =Conv(3,int(64*w),3,2,1)
        self.c2 =Conv(int(64*w),int(128*w),3,2,1)
        self.c2f1 = C2f(int(128*w),int(128*w),int(3*d),True)
        self.c3 = Conv(int(128*w),int(256*w),3,2,1)
        self.c2f2 = C2f(int(256*w),int(256*w),int(3*d),True)
        self.c4 = Conv(int(256*w),int(512*w),3,2,1)
        self.c2f3 = C2f(int(512*w),int(512*w),int(3*d),True)
        self.c5 = Conv(int(512*w),int(512*w*r),3,2,1)
        self.c2f4 = C2f(int(512*w*r),int(512*w*r),int(3*d),True)
        self.sppf =SPPF(int(512*w*r),int(512*w*r))

        self.c2f5 = C2f(int(512*w*(1+r)),int(512*w),int(3*d),False)
        self.c2f6 = C2f(int(768*w),int(256*w),int(3*d),False)

        self.c6 = Conv(int(256*w),int(256*w),3,2,1)
        self.c2f7 = C2f(int(768*w),int(512*w),int(3*d),False)
        self.c7 = Conv(int(512*w),int(512*w),3,2,1)
        self.c2f8 = C2f(int(512*w*(1+r)),int(512*w*r),int(3*d),False)

        self.upsample=nn.Upsample(scale_factor=(1,2,2))

        self.d1 = Detect(int(256*w))
        self.d1.bias_init(strides_[0])
        self.d2 = Detect(int(512*w))
        self.d2.bias_init(strides_[1])
        self.d3 = Detect(int(512*w*r))
        self.d3.bias_init(strides_[2])
        
        self.strides = None        
        self.anchors = None
        self.shape = None

        self.dfl = DFL(reg_max)
        self.dfl.requires_grad_(False)
        
    def forward(self,x):
        x = x.transpose(1,2)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c2f1(x)
        x = self.c3(x)
        x = self.c2f2(x)
        x1 = self.c4(x)
        x1 = self.c2f3(x1)
        x2 = self.c5(x1)
        x2 = self.c2f4(x2)
        x2 = self.sppf(x2)
        
        x3 = self.upsample(x2)
        x3 = torch.cat([x3,x1],dim=1)
        x3 = self.c2f5(x3)
    

        x4 = self.upsample(x3)
        x4 = torch.cat([x4,x],dim = 1)
        x4 = self.c2f6(x4)

        x5 = self.c6(x4)
        x5 = torch.cat([x5,x3],dim=1)
        x5 = self.c2f7(x5)

        x6 = self.c7(x5)
        x6 = torch.cat([x6,x2],dim = 1)
        x6 = self.c2f8(x6)

        x4 = self.d1(x4)
        x5 = self.d2(x5)
        x6 = self.d3(x6)


        x = [x4,x5,x6]
        if self.training:
            return x
        
        
        # Inference path
        x = [xi.flatten(end_dim=1) for xi in x]
        shape = x[0].shape  # BTCHW
        
        x_cat = torch.cat([xi.view(shape[0], num_of_cls+reg_max*4, -1) for xi in x], 2)
        
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, strides_, 0.5))
            self.shape = shape
        
        box, cls = x_cat.split((reg_max * 4, num_of_cls), 1)
        
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        
        return torch.cat((dbox, cls.sigmoid()), 1)
    