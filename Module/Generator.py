import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.autograd

from tensorboardX import SummaryWriter

from Module.GResBlock import GResBlock
from Module.Normalization import SpectralNorm
from Module.ConvGRU import ConvGRU
from Module.Attention import SelfAttention, SeparableAttn
# from Module.CrossReplicaBN import ScaledCrossReplicaBatchNorm2d
#from Module.ConvLSTM import ConvBLSTM, ConvLSTM

# class Binarization(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,x):
#         return torch.round(x)

#     def backward(self, x):
#         return x

# class Mask_Generator_Net(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#         ###### C-Block Stacking ##########
#         self.l11 = nn.Conv2d(in_channels, 64, 5, padding='same')
#         self.l12 = nn.Conv2d(in_channels, 32, 5, padding='same', dilation=2)
#         self.l13 = nn.Conv2d(in_channels, 32, 5, padding='same', dilation=5)
#         self.a11 = nn.ELU()
#         self.a12 = nn.ELU()
#         self.a13 = nn.ELU()
#         self.d1 = nn.MaxPool2d(2)
#         self.l21 = nn.Conv2d(128, 64, 5, padding='same')
#         self.l22 = nn.Conv2d(128, 32, 5, padding='same', dilation=2)
#         self.l23 = nn.Conv2d(128, 32, 5, padding='same', dilation=5)
#         self.a21 = nn.ELU()
#         self.a22 = nn.ELU()
#         self.a23 = nn.ELU()
#         self.d2 = nn.MaxPool2d(2)
#         self.l31 = nn.Conv2d(128, 128, 5, padding='same')
#         self.l32 = nn.Conv2d(128, 64, 5, padding='same', dilation=2)
#         self.l33 = nn.Conv2d(128, 64, 5, padding='same', dilation=5)
#         self.a31 = nn.ELU()
#         self.a32 = nn.ELU()
#         self.a33 = nn.ELU()
#         self.d3 = nn.MaxPool2d(2)
#         self.l41 = nn.Conv2d(256, 128, 5, padding='same')
#         self.l42 = nn.Conv2d(256, 64, 5, padding='same', dilation=2)
#         self.l43 = nn.Conv2d(256, 64, 5, padding='same', dilation=5)
#         self.a41 = nn.ELU()
#         self.a42 = nn.ELU()
#         self.a43 = nn.ELU()
#         self.d4 = nn.MaxPool2d(2)
#         self.l51 = nn.Conv2d(256, 256, 5, padding='same')
#         self.l52 = nn.Conv2d(256, 128, 5, padding='same', dilation=2)
#         self.l53 = nn.Conv2d(256, 128, 5, padding='same', dilation=5)
#         self.a51 = nn.ELU()
#         self.a52 = nn.ELU()
#         self.a53 = nn.ELU()
#         self.convBlstm = ConvBLSTM(512, 512, (5,5), 1, batch_first=True)
#         self.u1 = nn.Upsample(scale_factor=2)
#         ###### TC-Block Stacking ##########
#         self.tl11 = nn.ConvTranspose2d(256+512, 128, 5, padding=2)
#         self.tl12 = nn.ConvTranspose2d(256+512, 64, 5, dilation=2, padding=4)
#         self.tl13 = nn.ConvTranspose2d(256+512, 64, 5, dilation=5, padding=10)
#         self.ta11 = nn.ELU()
#         self.ta12 = nn.ELU()
#         self.ta13 = nn.ELU()
#         self.u2 = nn.Upsample(scale_factor=2)
#         self.tl21 = nn.ConvTranspose2d(256+256, 128, 5, padding=2)
#         self.tl22 = nn.ConvTranspose2d(256+256, 64, 5, dilation=2, padding=4)
#         self.tl23 = nn.ConvTranspose2d(256+256, 64, 5, dilation=5, padding=10)
#         self.ta21 = nn.ELU()
#         self.ta22 = nn.ELU()
#         self.ta23 = nn.ELU()
#         self.u3 = nn.Upsample(scale_factor=2)
#         self.tl31 = nn.ConvTranspose2d(256+128, 64, 5, padding=2)
#         self.tl32 = nn.ConvTranspose2d(256+128, 32, 5, padding=4, dilation=2)
#         self.tl33 = nn.ConvTranspose2d(256+128, 32, 5, padding=10, dilation=5)
#         self.ta31 = nn.ELU()
#         self.ta32 = nn.ELU()
#         self.ta33 = nn.ELU()
#         self.u4 = nn.Upsample(scale_factor=2)
#         self.tl41 = nn.ConvTranspose2d(128+128, 64, 5, padding=2)
#         self.tl42 = nn.ConvTranspose2d(128+128, 32, 5, padding=4, dilation=2)
#         self.tl43 = nn.ConvTranspose2d(128+128, 32, 5, padding=10, dilation=5)
#         self.ta41 = nn.ELU()
#         self.ta42 = nn.ELU()
#         self.ta43 = nn.ELU()
#         ###### Conv Layer Stacking ##########
#         self.conv1 = nn.ConvTranspose2d(128+in_channels,8, 3, padding=1)
#         self.conv2 = nn.ConvTranspose2d(8, 1, 3, padding=1)
#         self.conva1 = nn.ELU()
#         self.conva2 = nn.Hardsigmoid()
#         self.mask = Binarization()

#     def forward(self, x):
#         #### Conv Block 1
#         B,T,C,H,W = x.shape
#         x = x.view(B*T,C,H,W)
#         x11 = self.a11(self.l11(x))
#         x12 = self.a12(self.l12(x))
#         x13 = self.a13(self.l13(x))
#         x1o = torch.cat([x11,x12,x13], dim=1)
#         x1 = self.d1(x1o)
#         #### Conv Block 2
#         x21 = self.a21(self.l21(x1))
#         x22 = self.a22(self.l22(x1))
#         x23 = self.a23(self.l23(x1))
#         x2o = torch.cat([x21,x22,x23], dim=1)
#         x2 = self.d2(x2o)
#         #### Conv Block 3
#         x31 = self.a31(self.l31(x2))
#         x32 = self.a32(self.l32(x2))
#         x33 = self.a33(self.l33(x2))
#         x3o = torch.cat([x31,x32,x33], dim=1)
#         x3 = self.d3(x3o)
#         #### Conv Block 4
#         x41 = self.a41(self.l41(x3))
#         x42 = self.a42(self.l42(x3))
#         x43 = self.a43(self.l43(x3))
#         x4o = torch.cat([x41,x42,x43], dim=1)
#         x4 = self.d4(x4o)
#         #### Conv Block 5
#         x51 = self.a51(self.l51(x4))
#         x52 = self.a52(self.l52(x4))
#         x53 = self.a53(self.l53(x4))
#         x5o = torch.cat([x51,x52,x53], dim=1)
#         t,n,h,w = x5o.shape
#         x5 = x5o.view(1, t, n, h, w)
#         #### ConvBiLSTM
#         x_temp = self.convBlstm(x5,torch.flip(x5,[1]))
#         _, t,n,h,w = x_temp.shape
#         x_resized = x_temp.view(t,n,h,w)
#         x_lstm = self.u1(x_resized)       
#         #### Tr-Conv Block 1
#         y1c = torch.cat([x_lstm, x4o], dim=1)
#         y11 = self.ta11(self.tl11(y1c))
#         y12 = self.ta12(self.tl12(y1c))
#         y13 = self.ta13(self.tl13(y1c))
#         y1o = torch.cat([y11, y12, y13], dim=1)
#         y2 = self.u2(y1o)
#         #### Tr-Conv Block 2
#         y2c = torch.cat([y2, x3o], dim=1)
#         y21 = self.ta21(self.tl21(y2c))
#         y22 = self.ta22(self.tl22(y2c))
#         y23 = self.ta23(self.tl23(y2c))
#         y2o = torch.cat([y21, y22, y23], dim=1)
#         y3 = self.u3(y2o)
#         #### Tr-Conv Block 3
#         y3c = torch.cat([y3, x2o], dim=1)
#         y31 = self.ta31(self.tl31(y3c))
#         y32 = self.ta32(self.tl32(y3c))
#         y33 = self.ta33(self.tl33(y3c))
#         y3o = torch.cat([y31, y32, y33], dim=1)
#         y4 = self.u3(y3o)
#         #### Tr-Conv Block 4
#         y4c = torch.cat([y4, x1o], dim=1)
#         y41 = self.ta41(self.tl41(y4c))
#         y42 = self.ta42(self.tl42(y4c))
#         y43 = self.ta43(self.tl43(y4c))
#         y4o = torch.cat([y41, y42, y43, x], dim=1)
#         #### Conv Layer Head 1
#         y5 = self.conva1(self.conv1(y4o))
#         y6o = self.conva2(self.conv2(y5))
#         y6 = self.mask(y6o)
#         T,C,H,W = y6.shape
#         return y6.view(1,T,C,H,W)


# class Generator_Net(nn.Module):
#     def __init__(self, in_channels=6):
#         super().__init__()
#         ###### C-Block Stacking ##########
#         self.l11 = nn.Conv2d(in_channels, 64, 5, padding='same')
#         self.l12 = nn.Conv2d(in_channels, 32, 5, padding='same', dilation=2)
#         self.l13 = nn.Conv2d(in_channels, 32, 5, padding='same', dilation=5)
#         self.a11 = nn.ELU()
#         self.a12 = nn.ELU()
#         self.a13 = nn.ELU()
#         self.d1 = nn.MaxPool2d(2)
#         self.l21 = nn.Conv2d(128, 64, 5, padding='same')
#         self.l22 = nn.Conv2d(128, 32, 5, padding='same', dilation=2)
#         self.l23 = nn.Conv2d(128, 32, 5, padding='same', dilation=5)
#         self.a21 = nn.ELU()
#         self.a22 = nn.ELU()
#         self.a23 = nn.ELU()
#         self.d2 = nn.MaxPool2d(2)
#         self.l31 = nn.Conv2d(128, 128, 5, padding='same')
#         self.l32 = nn.Conv2d(128, 64, 5, padding='same', dilation=2)
#         self.l33 = nn.Conv2d(128, 64, 5, padding='same', dilation=5)
#         self.a31 = nn.ELU()
#         self.a32 = nn.ELU()
#         self.a33 = nn.ELU()
#         self.d3 = nn.MaxPool2d(2)
#         self.l41 = nn.Conv2d(256, 128, 5, padding='same')
#         self.l42 = nn.Conv2d(256, 64, 5, padding='same', dilation=2)
#         self.l43 = nn.Conv2d(256, 64, 5, padding='same', dilation=5)
#         self.a41 = nn.ELU()
#         self.a42 = nn.ELU()
#         self.a43 = nn.ELU()
#         self.d4 = nn.MaxPool2d(2)
#         self.l51 = nn.Conv2d(256, 256, 5, padding='same')
#         self.l52 = nn.Conv2d(256, 128, 5, padding='same', dilation=2)
#         self.l53 = nn.Conv2d(256, 128, 5, padding='same', dilation=5)
#         self.a51 = nn.ELU()
#         self.a52 = nn.ELU()
#         self.a53 = nn.ELU()
#         self.convBlstm = ConvBLSTM(512, 512, (5,5), 1, batch_first=True)
#         self.u1 = nn.Upsample(scale_factor=2)
#         ###### TC-Block Stacking ##########
#         self.tl11 = nn.ConvTranspose2d(256+512, 128, 5, padding=2)
#         self.tl12 = nn.ConvTranspose2d(256+512, 64, 5, dilation=2, padding=4)
#         self.tl13 = nn.ConvTranspose2d(256+512, 64, 5, dilation=5, padding=10)
#         self.ta11 = nn.ELU()
#         self.ta12 = nn.ELU()
#         self.ta13 = nn.ELU()
#         self.u2 = nn.Upsample(scale_factor=2)
#         self.tl21 = nn.ConvTranspose2d(256+256, 128, 5, padding=2)
#         self.tl22 = nn.ConvTranspose2d(256+256, 64, 5, dilation=2, padding=4)
#         self.tl23 = nn.ConvTranspose2d(256+256, 64, 5, dilation=5, padding=10)
#         self.ta21 = nn.ELU()
#         self.ta22 = nn.ELU()
#         self.ta23 = nn.ELU()
#         self.u3 = nn.Upsample(scale_factor=2)
#         self.tl31 = nn.ConvTranspose2d(256+128, 64, 5, padding=2)
#         self.tl32 = nn.ConvTranspose2d(256+128, 32, 5, padding=4, dilation=2)
#         self.tl33 = nn.ConvTranspose2d(256+128, 32, 5, padding=10, dilation=5)
#         self.ta31 = nn.ELU()
#         self.ta32 = nn.ELU()
#         self.ta33 = nn.ELU()
#         self.u4 = nn.Upsample(scale_factor=2)
#         self.tl41 = nn.ConvTranspose2d(128+128, 64, 5, padding=2)
#         self.tl42 = nn.ConvTranspose2d(128+128, 32, 5, padding=4, dilation=2)
#         self.tl43 = nn.ConvTranspose2d(128+128, 32, 5, padding=10, dilation=5)
#         self.ta41 = nn.ELU()
#         self.ta42 = nn.ELU()
#         self.ta43 = nn.ELU()
#         ###### Conv Layer Stacking ##########
#         self.conv1 = nn.ConvTranspose2d(128+in_channels,8, 3, padding=1)
#         self.conv2 = nn.ConvTranspose2d(8, 3, 3, padding=1)
#         self.conva1 = nn.ELU()
#         self.conva2 = nn.Hardsigmoid()

#     def forward(self, x):
#         B,T,C,H,W = x.shape
#         x = x.view(B*T,C,H,W)
#         #### Conv Block 1
#         x11 = self.a11(self.l11(x))
#         x12 = self.a12(self.l12(x))
#         x13 = self.a13(self.l13(x))
#         x1o = torch.cat([x11,x12,x13], dim=1)
#         x1 = self.d1(x1o)
#         #### Conv Block 2
#         x21 = self.a21(self.l21(x1))
#         x22 = self.a22(self.l22(x1))
#         x23 = self.a23(self.l23(x1))
#         x2o = torch.cat([x21,x22,x23], dim=1)
#         x2 = self.d2(x2o)
#         #### Conv Block 3
#         x31 = self.a31(self.l31(x2))
#         x32 = self.a32(self.l32(x2))
#         x33 = self.a33(self.l33(x2))
#         x3o = torch.cat([x31,x32,x33], dim=1)
#         x3 = self.d3(x3o)
#         #### Conv Block 4
#         x41 = self.a41(self.l41(x3))
#         x42 = self.a42(self.l42(x3))
#         x43 = self.a43(self.l43(x3))
#         x4o = torch.cat([x41,x42,x43], dim=1)
#         x4 = self.d4(x4o)
#         #### Conv Block 5
#         x51 = self.a51(self.l51(x4))
#         x52 = self.a52(self.l52(x4))
#         x53 = self.a53(self.l53(x4))
#         x5o = torch.cat([x51,x52,x53], dim=1)
#         t,n,h,w = x5o.shape
#         x5 = x5o.view(1, t, n, h, w)
#         #### ConvBiLSTM
#         x_temp = self.convBlstm(x5,torch.flip(x5,[1]))
#         _, t,n,h,w = x_temp.shape
#         x_resized = x_temp.view(t,n,h,w)
#         x_lstm = self.u1(x_resized)       
#         #### Tr-Conv Block 1
#         y1c = torch.cat([x_lstm, x4o], dim=1)
#         y11 = self.ta11(self.tl11(y1c))
#         y12 = self.ta12(self.tl12(y1c))
#         y13 = self.ta13(self.tl13(y1c))
#         y1o = torch.cat([y11, y12, y13], dim=1)
#         y2 = self.u2(y1o)
#         #### Tr-Conv Block 2
#         y2c = torch.cat([y2, x3o], dim=1)
#         y21 = self.ta21(self.tl21(y2c))
#         y22 = self.ta22(self.tl22(y2c))
#         y23 = self.ta23(self.tl23(y2c))
#         y2o = torch.cat([y21, y22, y23], dim=1)
#         y3 = self.u3(y2o)
#         #### Tr-Conv Block 3
#         y3c = torch.cat([y3, x2o], dim=1)
#         y31 = self.ta31(self.tl31(y3c))
#         y32 = self.ta32(self.tl32(y3c))
#         y33 = self.ta33(self.tl33(y3c))
#         y3o = torch.cat([y31, y32, y33], dim=1)
#         y4 = self.u3(y3o)
#         #### Tr-Conv Block 4
#         y4c = torch.cat([y4, x1o], dim=1)
#         y41 = self.ta41(self.tl41(y4c))
#         y42 = self.ta42(self.tl42(y4c))
#         y43 = self.ta43(self.tl43(y4c))
#         y4o = torch.cat([y41, y42, y43, x], dim=1)
#         #### Conv Layer Head 1
#         y5 = self.conva1(self.conv1(y4o))
#         y6 = self.conva2(self.conv2(y5))
#         T,C,H,W = y6.shape
#         return y6.view(1,T,C,H,W)


class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.round(input)

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

class Mask_Generator_Net(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        ###### C-Block Stacking ##########
        self.l11 = nn.Conv2d(in_channels, 32, 5, padding='same')
        self.l12 = nn.Conv2d(in_channels, 32, 5, padding='same', dilation=2)
        self.d1 = nn.MaxPool2d(2)
        self.l21 = nn.Conv2d(64, 64, 5, padding='same')
        self.l22 = nn.Conv2d(64, 64, 5, padding='same', dilation=2)
        self.d2 = nn.MaxPool2d(2)
        self.l31 = nn.Conv2d(128, 128, 5, padding='same')
        self.l32 = nn.Conv2d(128, 128, 5, padding='same', dilation=2)
        self.d3 = nn.MaxPool2d(2)
        self.convgru = ConvGRU(256, hidden_sizes=[512], kernel_sizes=[5], n_layers=1)
        self.u1 = nn.Upsample(scale_factor=2)
        self.tl11 = nn.ConvTranspose2d(256+512, 128, 5, padding=2)
        self.tl12 = nn.ConvTranspose2d(256+512, 128, 5, dilation=2, padding=4)
        self.u2 = nn.Upsample(scale_factor=2)
        self.tl21 = nn.ConvTranspose2d(128+256, 64, 5, padding=2)
        self.tl22 = nn.ConvTranspose2d(128+256, 64, 5, padding=4, dilation=2)
        self.u3 = nn.Upsample(scale_factor=2)
        self.tl31 = nn.ConvTranspose2d(64+128, 32, 5, padding=2)
        self.tl32 = nn.ConvTranspose2d(64+128, 32, 5, padding=4, dilation=2)
        ###### Conv Layer Stacking ##########
        self.conv1 = nn.ConvTranspose2d(64+in_channels,8, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(8, 1, 3, padding=1)
        self.mask = Binarization.apply

    def forward(self, x):
        #### Conv Block 1
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        temp1 = F.elu(self.l11(x))
        temp2 = F.elu(self.l12(x))
        x1o = torch.cat([temp1,temp2], dim=1)
        out = self.d1(x1o)
        
        #### Conv Block 2
        temp1 = F.elu(self.l21(out))
        temp2 = F.elu(self.l22(out))
        x2o = torch.cat([temp1,temp2], dim=1)
        out = self.d2(x2o)
        
        #### Conv Block 3
        temp1 = F.elu(self.l31(out))
        temp2 = F.elu(self.l32(out))
        x3o = torch.cat([temp1,temp2], dim=1)
        out = self.d3(x3o)
        
        # #### Conv Block 4
        # temp1 = F.elu(self.l51(out))
        # temp2 = F.elu(self.l52(out))
        # x5o = torch.cat([temp1,temp2], dim=1)
        t,n,h,w = out.shape
        out = out.view(B, T, n, h, w)
        #### ConvBiLSTM
        frame_list = []
        for i in range(T):
            if i == 0:
                frame_list.append(self.convgru(out[:,0,:,:,:].squeeze(1)))  # T x [B x ch x ld x ld]
            else:
                frame_list.append(self.convgru(out[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
        frame_hidden_list = []
        for i in frame_list:
            frame_hidden_list.append(i[-1].unsqueeze(0))
        out = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

        out = out.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
        B, T, C, W, H = out.size()
        out = out.view(-1, C, W, H)
        out = self.u1(out) 
        
        #### Tr-Conv Block 2
        out = torch.cat([out, x3o], dim=1)
        temp1 = F.elu(self.tl11(out))
        temp2 = F.elu(self.tl12(out))
        out = torch.cat([temp1, temp2], dim=1)
        out = self.u2(out)
        
        #### Tr-Conv Block 3
        out = torch.cat([out, x2o], dim=1)
        temp1 = F.elu(self.tl21(out))
        temp2 = F.elu(self.tl22(out))
        out = torch.cat([temp1, temp2], dim=1)
        out = self.u3(out)
        
        #### Tr-Conv Block 4
        out = torch.cat([out, x1o], dim=1)
        temp1 = F.elu(self.tl31(out))
        temp2 = F.elu(self.tl32(out))
        out = torch.cat([temp1, temp2, x], dim=1)
        
        #### Conv Layer Head 1
        out = F.elu(self.conv1(out))
        
        out = F.hardsigmoid(self.conv2(out))
        
        out = self.mask(out)
        T,C,H,W = out.shape
        return out.view(B,-1,C,H,W)

class Generator_Net(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        ###### C-Block Stacking ##########
        self.l11 = nn.Conv2d(in_channels, 32, 5, padding='same')
        self.l12 = nn.Conv2d(in_channels, 32, 5, padding='same', dilation=2)
        self.d1 = nn.MaxPool2d(2)
        self.l21 = nn.Conv2d(64, 64, 5, padding='same')
        self.l22 = nn.Conv2d(64, 64, 5, padding='same', dilation=2)
        self.d2 = nn.MaxPool2d(2)
        self.l31 = nn.Conv2d(128, 128, 5, padding='same')
        self.l32 = nn.Conv2d(128, 128, 5, padding='same', dilation=2)
        self.d3 = nn.MaxPool2d(2)
        self.convgru = ConvGRU(256, hidden_sizes=[512], kernel_sizes=[5], n_layers=1)
        self.u1 = nn.Upsample(scale_factor=2)
        self.tl11 = nn.ConvTranspose2d(256+512, 128, 5, padding=2)
        self.tl12 = nn.ConvTranspose2d(256+512, 128, 5, dilation=2, padding=4)
        self.u2 = nn.Upsample(scale_factor=2)
        self.tl21 = nn.ConvTranspose2d(128+256, 64, 5, padding=2)
        self.tl22 = nn.ConvTranspose2d(128+256, 64, 5, padding=4, dilation=2)
        self.u3 = nn.Upsample(scale_factor=2)
        self.tl31 = nn.ConvTranspose2d(64+128, 32, 5, padding=2)
        self.tl32 = nn.ConvTranspose2d(64+128, 32, 5, padding=4, dilation=2)
        ###### Conv Layer Stacking ##########
        self.conv1 = nn.ConvTranspose2d(64+in_channels,8, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(8, 3, 3, padding=1)

    def forward(self, x):
        #### Conv Block 1
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        temp1 = F.elu(self.l11(x))
        temp2 = F.elu(self.l12(x))
        x1o = torch.cat([temp1,temp2], dim=1)
        out = self.d1(x1o)
        
        #### Conv Block 2
        temp1 = F.elu(self.l21(out))
        temp2 = F.elu(self.l22(out))
        x2o = torch.cat([temp1,temp2], dim=1)
        out = self.d2(x2o)
        
        #### Conv Block 3
        temp1 = F.elu(self.l31(out))
        temp2 = F.elu(self.l32(out))
        x3o = torch.cat([temp1,temp2], dim=1)
        out = self.d3(x3o)
        
        # #### Conv Block 4
        # temp1 = F.elu(self.l51(out))
        # temp2 = F.elu(self.l52(out))
        # x5o = torch.cat([temp1,temp2], dim=1)
        t,n,h,w = out.shape
        out = out.view(B, T, n, h, w)
        #### ConvBiLSTM
        frame_list = []
        for i in range(T):
            if i == 0:
                frame_list.append(self.convgru(out[:,0,:,:,:].squeeze(1)))  # T x [B x ch x ld x ld]
            else:
                frame_list.append(self.convgru(out[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
        frame_hidden_list = []
        for i in frame_list:
            frame_hidden_list.append(i[-1].unsqueeze(0))
        out = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

        out = out.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
        B, T, C, W, H = out.size()
        out = out.view(-1, C, W, H)
        out = self.u1(out) 
        
        #### Tr-Conv Block 2
        out = torch.cat([out, x3o], dim=1)
        temp1 = F.elu(self.tl11(out))
        temp2 = F.elu(self.tl12(out))
        out = torch.cat([temp1, temp2], dim=1)
        out = self.u2(out)
        
        #### Tr-Conv Block 3
        out = torch.cat([out, x2o], dim=1)
        temp1 = F.elu(self.tl21(out))
        temp2 = F.elu(self.tl22(out))
        out = torch.cat([temp1, temp2], dim=1)
        out = self.u3(out)
        
        #### Tr-Conv Block 4
        out = torch.cat([out, x1o], dim=1)
        temp1 = F.elu(self.tl31(out))
        temp2 = F.elu(self.tl32(out))
        out = torch.cat([temp1, temp2, x], dim=1)
        
        #### Conv Layer Head 1
        out = F.elu(self.conv1(out))
        
        out = F.hardsigmoid(self.conv2(out))
        
        T,C,H,W = out.shape
        return out.view(B,-1,C,H,W)

class Generator(nn.Module):

    def __init__(self, in_dim=120, latent_dim=4, n_class=4, ch=32, n_frames=48, hierar_flag=False):
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_class = n_class
        self.ch = ch
        self.hierar_flag = hierar_flag
        self.n_frames = n_frames

        self.embedding = nn.Embedding(n_class, in_dim)

        self.affine_transfrom = nn.Linear(in_dim * 2, latent_dim * latent_dim * 8 * ch)

        # self.self_attn = SelfAttention(8 * ch)

        # self.conv = nn.ModuleList([GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
        #                            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
        #                            GResBlock(8 * ch, 4 * ch, n_class=in_dim * 2),
        #                            SeparableAttn(4 * ch),
        #                            GResBlock(4 * ch, 2 * ch, n_class=in_dim * 2)])
        # self.convGRU = ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3)

        self.conv = nn.ModuleList([
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(8 * ch, 4 * ch, n_class=in_dim * 2),
            ConvGRU(4 * ch, hidden_sizes=[4 * ch, 8 * ch, 4 * ch], kernel_sizes=[3, 5, 5], n_layers=3),
            # ConvGRU(4 * ch, hidden_sizes=[4 * ch, 4 * ch], kernel_sizes=[3, 5], n_layers=2),
            GResBlock(4 * ch, 4 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(4 * ch, 2 * ch, n_class=in_dim * 2)
        ])

        # TODO impl ScaledCrossReplicaBatchNorm
        # self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1 * chn)

        self.colorize = SpectralNorm(nn.Conv2d(2 * ch, 3, kernel_size=(3, 3), padding=1))


    def forward(self, x, class_id):

        if self.hierar_flag is True:
            noise_emb = torch.split(x, self.in_dim, dim=1)
        else:
            noise_emb = x

        class_emb = self.embedding(class_id)

        if self.hierar_flag is True:
            y = self.affine_transfrom(torch.cat((noise_emb[0], class_emb), dim=1)) # B x (2 x ld x ch)
        else:
            y = self.affine_transfrom(torch.cat((noise_emb, class_emb), dim=1)) # B x (2 x ld x ch)

        y = y.view(-1, 8 * self.ch, self.latent_dim, self.latent_dim) # B x ch x ld x ld

        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):

                if k > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.n_frames, C, W, H).contiguous()

                frame_list = []
                for i in range(self.n_frames):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:,0,:,:,:].squeeze(1)))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

                y = y.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
                # print(y.size())
                B, T, C, W, H = y.size()
                y = y.view(-1, C, W, H)

            elif isinstance(conv, GResBlock):
                condition = torch.cat([noise_emb, class_emb], dim=1)
                condition = condition.repeat(self.n_frames,1)
                y = conv(y, condition) # BT, C, W, H

        y = F.relu(y)
        y = self.colorize(y)
        y = torch.tanh(y)

        BT, C, W, H = y.size()
        y = y.view(-1, self.n_frames, C, W, H) # B, T, C, W, H

        return y

        # if torch.cuda.is_available():
        #     frame_list = torch.empty((0,)).cuda()  # initialization similar to frame_list = []
        # else:
        #     frame_list = torch.empty((0,))
        #
        # for i in range(self.n_frames):
        #     if i == 0:
        #         frame_list = self.convGRU(y) # T x [B x ch x ld x ld]
        #     else:
        #         frame_list = torch.stack((frame_list, self.convGRU(y, frame_list[i-1])))
        #
        # frame_hidden_list = torch.empty((0,)).cuda()
        # for i in frame_list.size()[0]:
        #     if i == 0:
        #         frame_hidden_list = 1
        #     frame_hidden_list.append(i[-1].unsqueeze(0))

        # frame_list = []
        # for i in range(self.n_frames):
        #     if i == 0:
        #         frame_list.append(self.convGRU(y))  # T x [B x ch x ld x ld]
        #     else:
        #         frame_list.append(self.convGRU(y, frame_list[i - 1]))
        #
        # frame_hidden_list = []
        # for i in frame_list:
        #     frame_hidden_list.append(i[-1].unsqueeze(0))
        # y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld
        # y = y.permute(1, 2, 0, 3, 4) # B x ch x T x ld x ld
        # y = self.self_attn(y)  # B x ch x T x ld x ld
        #
        # y = y.permute(0, 2, 1, 3, 4).contiguous() # B x T x ch x ld x ld
        #
        # # the time axis is folded into the batch axis before the forward pass, which applying ResNet to all frames indivudually
        # y = y.view(-1, 8 * self.ch, self.latent_dim, self.latent_dim) # (B x T) x ch x ld x ld
        #
        # frame = y
        # for j, conv in enumerate(self.conv):
        #     if isinstance(conv, GResBlock):
        #         condition = torch.cat([noise_emb, class_emb], dim=1)
        #         condition = condition.repeat(self.n_frames, 1)
        #         frame = conv(frame, condition)
        #     else:
        #         BT, C, W, H = frame.size()
        #         frame = frame.view(-1, self.n_frames, C, W, H).transpose(2, 1) # B, C, T, W, H
        #         frame = conv(frame)
        #         frame = frame.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, W, H) # BT, C, W, H
        #
        # frame = F.relu(frame)
        # frame = self.colorize(frame)
        # frame = torch.tanh(frame)
        #
        # BT, C, W, H = frame.size()
        # frame = frame.view(-1, self.n_frames, C, W, H) # B, T, C, W, H

        # return frame


if __name__ == "__main__":

    batch_size = 5
    in_dim = 120
    n_class = 4
    n_frames = 4

    x = torch.randn(batch_size, in_dim).cuda()
    class_label = torch.randint(low=0, high=3, size=(batch_size,)).cuda()
    generator = Generator(in_dim, n_class=n_class, ch=3, n_frames=n_frames).cuda()
    y = generator(x, class_label)

    print(x.size())
    print(y.size())
