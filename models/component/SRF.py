import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import einops



class EdgeConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 bias=True):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1,
            bias=bias)
        self.w_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(1, kernel_size), 
            stride=1, 
            padding=(0, kernel_size//2),
            groups=mid_channels)
        
        self.h_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(kernel_size, 1), 
            stride=1, 
            padding=(kernel_size//2, 0),
            groups=mid_channels
        )
        
        self.out_proj = nn.Conv2d(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x


class EDTM(nn.Module):
    def __init__(self,
                 in_dim,
                 nbins,
                 cell_size = (8, 8)):
        super().__init__()

        self.nbins = nbins
        self.cell_size = cell_size

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.ReLU(inplace=True),  
            nn.AdaptiveAvgPool2d((1, 1))   
        )


        self.weight = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim//2, out_channels=in_dim),
            nn.GroupNorm(in_dim//8, in_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim//8, in_dim)
        )

        self.fuse_block = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim//2, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(in_dim//8, in_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        x = image2patches(x)

        x_hog = self.get_hog_feature(x)
        x_hog = self.hog_feat(x_hog)

        x1 = self.sigmoid(self.weight(x + x_hog))
        x2 = self.conv(x)
        x = x1 * x2

        x = patches2image(x)

        x = x + residual
        x = self.fuse_block(x)

        return x



    def get_hog_feature(self, x):
        
        x_mean = x.mean(dim=1, keepdim=True)
        B, _, H, W = x_mean.shape
        device = x_mean.device

        sobel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 
                            dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], 
                            dtype=torch.float32).view(1, 1, 3, 3).to(device)
        dx = F.conv2d(x_mean.float(), sobel_x, padding=1)  # b, 1, h, w
        dy = F.conv2d(x_mean.float(), sobel_y, padding=1)
        
        # direction
        gradient_dir = torch.atan2(dy, dx)       # [-π，π] 
        gradient_dir = torch.abs(gradient_dir)   # [0，π] 
        
        # cells
        cell_h, cell_w = self.cell_size
        H_cells = int(H / cell_h)
        W_cells = int(W / cell_w)

        # 
        dirs_crop = gradient_dir[:, :, :H_cells*cell_h, :W_cells*cell_w]

        # [B, H_cells, W_cells, cell_h*cell*w]
        dirs = dirs_crop.reshape(B, H_cells, W_cells, -1)
        
        bin_with = torch.pi / self.nbins
        bin_indices = (dirs / bin_with).floor().long() 
        bin_indices = torch.clamp(bin_indices, 0, self.nbins-1) 

        # Vectorized histogram counting over all cells to avoid Python-side loops.
        weight = F.one_hot(bin_indices, num_classes=self.nbins).sum(dim=3).float()
        weight = weight /  64  # B, H_cells, W_cells, self.nbins 64

        start = torch.pi / (2*self.nbins)
        hog_feature = torch.linspace(start, torch.pi - start, self.nbins).to(device).repeat(B, H_cells, W_cells,1) * weight

        return hog_feature.permute(0, 3, 1, 2)
    


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = einops.rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = einops.rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x



class Conv2d_GN(nn.Module):
    def __init__(self, inc, outc, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, ks, stride, pad, dilation, groups, bias),
            nn.GroupNorm(outc//8, outc)
        )
    def forward(self, x):
        return self.block(x)



class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x
    

class SRModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.att = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2, x3, x4):

        _, _, H, W = x1.shape

        att_map = self.sigmoid(self.att(x1))

        x2 = nn.Upsample(size=(H, W), mode="bilinear")(x2)
        x3 = nn.Upsample(size=(H, W), mode="bilinear")(x3)
        x4 = nn.Upsample(size=(H, W), mode="bilinear")(x4)  

        x2 = x2 * att_map
        x3 = x3 * att_map
        x4 = x4 * att_map

        return x2, x3, x4


class SRFModule(nn.Module):
    def __init__(self, embed_dims, mid_dim=8, size=(512,512)):
        super(SRFModule, self).__init__()
        
        self.size = size

        self.mlp1 = MLP(input_dim=embed_dims[0], embed_dim=mid_dim)
        self.mlp2 = MLP(input_dim=embed_dims[1], embed_dim=mid_dim)
        self.mlp3 = MLP(input_dim=embed_dims[2], embed_dim=mid_dim)
        self.mlp4 = MLP(input_dim=embed_dims[3], embed_dim=mid_dim)

        self.conv1 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv2 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv3 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv4 = Conv2d_GN(inc=mid_dim, outc=mid_dim)



        self.sr_module = SRModule(dim=mid_dim)


        self.block = nn.Sequential(
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim*4),
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim),
            nn.ReLU()
        )


        self.linear_pred = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim//4, 1),
            nn.Conv2d(mid_dim//4, mid_dim//4, 1 ,groups=mid_dim//4),
            nn.Conv2d(mid_dim//4, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        

    def forward(self, inputs):
 
        
        x1, x2, x3, x4 = inputs # [16, 128, 128] [32, 64, 64] [64, 32, 32] [128, 16, 16]

        x1 = self.conv1(self.mlp1(x1))
        x2 = self.conv2(self.mlp2(x2))
        x3 = self.conv3(self.mlp3(x3))
        x4 = self.conv4(self.mlp4(x4))

        x2, x3, x4 = self.sr_module(x1, x2, x3, x4)  


        #### att_map vis 
        self.outs = [x1, x2, x3, x4]

        x1 = nn.Upsample(size=self.size, mode="bilinear")(x1)
        x2 = nn.Upsample(size=self.size, mode="bilinear")(x2)
        x3 = nn.Upsample(size=self.size, mode="bilinear")(x3)
        x4 = nn.Upsample(size=self.size, mode="bilinear")(x4)  #  [1, 16, 512, 512]

        x = self.block(torch.cat([x1,x2,x3,x4], dim=1))   # [1, 8, 512, 512]
        # x = torch.cat([x1+res, x2+res, x3+res, x4+res], dim=1)
        x = self.linear_pred(x)

        return x


class simple_SRModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.att = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2):

        _, _, H, W = x1.shape

        att_map = self.sigmoid(self.att(x1))

        x2 = nn.Upsample(size=(H, W), mode="bilinear")(x2)

        x2 = x2 * att_map

        return x2


class simple_SRFModule(nn.Module):
    def __init__(self, embed_dims, mid_dim=8, size=(512,512)):
        super(simple_SRFModule, self).__init__()
        
        self.size = size

        self.mlp1 = MLP(input_dim=embed_dims[0], embed_dim=mid_dim)
        self.mlp2 = MLP(input_dim=embed_dims[1], embed_dim=mid_dim)


        self.conv1 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv2 = Conv2d_GN(inc=mid_dim, outc=mid_dim)

        self.sr_module = simple_SRModule(dim=mid_dim)


        self.block = nn.Sequential(
            Conv2d_GN(inc=mid_dim*2, outc=mid_dim*2),
            Conv2d_GN(inc=mid_dim*2, outc=mid_dim),
            nn.ReLU()
        )


        self.linear_pred = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim//4, 1),
            nn.Conv2d(mid_dim//4, mid_dim//4, 1 ,groups=mid_dim//4),
            nn.Conv2d(mid_dim//4, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        

    def forward(self, inputs):
 
        
        x1, x2= inputs # [16, 128, 128] [32, 64, 64] [64, 32, 32] [128, 16, 16]

        x1 = self.conv1(self.mlp1(x1))

        x2 = self.conv2(self.mlp2(x2))

        x2= self.sr_module(x1, x2)  


        #### att_map vis 
        self.outs = [x1, x2]

        x2 = nn.Upsample(size=self.size, mode="bilinear")(x2)


        x = self.block(torch.cat([x1,x2], dim=1))   
        # x = torch.cat([x1+res, x2+res], dim=1)
        x = self.linear_pred(x)

        return x
    
class CSRModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.att = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2, x3, x4):

        _, _, H1, W1 = x1.shape
        _, _, H2, W2 = x3.shape
        att_map1 = self.sigmoid(self.att(x1))
        att_map2 = self.sigmoid(self.att(x3))
        x2 = nn.Upsample(size=(H1, W1), mode="bilinear")(x2)

        x4 = nn.Upsample(size=(H2, W2), mode="bilinear")(x4)

        x2 = x2 * att_map1
        x4 = x4 * att_map2

        return x2, x3, x4

class MGFFM(nn.Module):
    def __init__(self, embed_dims, mid_dim=8, size=(512,512)):
        super(MGFFM, self).__init__()
        
        self.size = size

        self.mlp1 = MLP(input_dim=embed_dims[0], embed_dim=mid_dim)
        self.mlp2 = MLP(input_dim=embed_dims[1], embed_dim=mid_dim)
        self.mlp3 = MLP(input_dim=embed_dims[2], embed_dim=mid_dim)
        self.mlp4 = MLP(input_dim=embed_dims[3], embed_dim=mid_dim)

        self.conv1 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv2 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv3 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv4 = Conv2d_GN(inc=mid_dim, outc=mid_dim)



        self.sr_module = SRModule(dim=mid_dim)


        self.block = nn.Sequential(
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim*4),
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim),
            nn.ReLU()
        )


        self.linear_pred = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim//4, 1),
            nn.Conv2d(mid_dim//4, mid_dim//4, 1 ,groups=mid_dim//4),
            nn.Conv2d(mid_dim//4, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        

    def forward(self, inputs):
 
        
        x1, x2, x3, x4 = inputs # 

        x1 = self.conv1(self.mlp1(x1))
        x2 = self.conv2(self.mlp2(x2))
        x3 = self.conv3(self.mlp3(x3))
        x4 = self.conv4(self.mlp4(x4))

        x2, x3, x4 = self.sr_module(x1, x2, x3, x4)  


        #### att_map vis 
        self.outs = [x1, x2, x3, x4]

        x1 = nn.Upsample(size=self.size, mode="bilinear")(x1)
        x2 = nn.Upsample(size=self.size, mode="bilinear")(x2)
        x3 = nn.Upsample(size=self.size, mode="bilinear")(x3)
        x4 = nn.Upsample(size=self.size, mode="bilinear")(x4)  

        x = self.block(torch.cat([x1,x2,x3,x4], dim=1))   # [1, 8, 512, 512]
        # x = torch.cat([x1+res, x2+res, x3+res, x4+res], dim=1)
        x_map = self.linear_pred(x)

        return x_map

