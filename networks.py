#!/usr/bin/env python3.6


import math

import torch
from torch import nn
from torch import Tensor

from layers import upSampleConv, conv_block_1, conv_block_3_3, conv_block_Asym
from layers import conv_block, conv_block_3, maxpool, conv_decod_block


from layers import upSampleConv, convBatch, residualConv

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Dummy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.down = nn.Conv2d(in_dim, 10, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(10, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, input: Tensor) -> Tensor:
        return self.up(self.down(input))

Dimwit = Dummy

class CoordConvDummy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.down = CoordConv2d(in_dim, 10, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(10, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, input: Tensor) -> Tensor:
        return self.up(self.down(input))

class UNet(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        bridge = self.bridge(x2)

        y0 = self.deconv1(bridge)
        y1 = self.deconv2(self.conv5(torch.cat((y0, x2), dim=1)))
        y2 = self.deconv3(self.conv6(torch.cat((y1, x1), dim=1)))
        y3 = self.conv7(torch.cat((y2, x0), dim=1))

        return self.final(y3)


class CoordConvUNet(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.conv0 = nn.Sequential(convBatch(nin, nG,layer=CoordConv2d ),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        bridge = self.bridge(x2)

        y0 = self.deconv1(bridge)
        y1 = self.deconv2(self.conv5(torch.cat((y0, x2), dim=1)))
        y2 = self.deconv3(self.conv6(torch.cat((y1, x1), dim=1)))
        y3 = self.conv7(torch.cat((y2, x0), dim=1))

        return self.final(y3)







import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x

class BBConv(Module):
    def __init__(self, in_feat, out_feat, pool_ratio):
        super(BBConv, self).__init__()
        self.mp = nn.MaxPool2d(pool_ratio)
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = F.sigmoid(x)
        return x


class clDUnet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self,n_channels,n_organs, drop_rate=0, bn_momentum=0.1):
        super(clDUnet, self).__init__()

        #Downsampling path
        self.conv1 = DownConv(n_channels, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        self.conv4 = DownConv(256,512, drop_rate, bn_momentum)
        self.mp4 = nn.MaxPool2d(2)
        
        self.conv5 = DownConv(512,1024, drop_rate, bn_momentum)
        self.mp5 = nn.MaxPool2d(2)


        # Bottle neck
        self.conv6 = DownConv(1024,1024, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(2048, 1024, drop_rate, bn_momentum)
        self.up2 = UpConv(1536, 512, drop_rate, bn_momentum)
        self.up3 = UpConv(768, 256, drop_rate, bn_momentum)
        self.up4 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up5 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, n_organs, kernel_size=3, padding=1)

    def forward(self, x, comment = ' '):
        x1 = self.conv1(x)
        p1 = self.mp1(x1)

        x2 = self.conv2(p1)
        p2 = self.mp2(x2)

        x3 = self.conv3(p2)
        p3 = self.mp3(x3)

        # Bottom
        x4 = self.conv4(p3)
        p4 = self.mp4(x4)
        
        x5 = self.conv5(p4)
        p5 = self.mp5(x5)
        
        x6 = self.conv6(p5)
    
        # Up-sampling
        u1 = self.up1(x6, x5)
        u2 = self.up2(u1, x4)
        u3 = self.up3(u2, x3)
        u4 = self.up4(u3, x2)
        u5 = self.up5(u4, x1)



        x6 = self.conv9(u5)
       
        return x6
class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3




def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super().__init__()

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
        # Secondary branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Main branch
        maxpool_output, indices = self.maxpool0(input)

        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        # Zero padding the feature maps from the main branch
        depth_to_pad = abs(maxpool_output.shape[1] - do.shape[1])
        padding = torch.zeros(maxpool_output.shape[0], depth_to_pad, maxpool_output.shape[2],
                              maxpool_output.shape[3], device=maxpool_output.device)
        maxpool_output_pad = torch.cat((maxpool_output, padding), 1)
        output = maxpool_output_pad + do

        # _, c, _, _ = maxpool_output.shape
        # output = do
        # output[:, :c, :, :] += maxpool_output

        final_output = self.PReLU3(output)

        return final_output, indices


class BottleNeckDownSamplingDilatedConv(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConv, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = input + do
        output = self.PReLU3(output)

        return output


class BottleNeckDownSamplingDilatedConvLast(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim, dilation):
        super(BottleNeckDownSamplingDilatedConvLast, self).__init__()
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):

        # Secondary branch
        b0 = self.block0(input)

        c1 = self.conv1(b0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        b2 = self.block2(p1)

        do = self.do(b2)

        output = self.conv_out(input) + do
        output = self.PReLU3(output)

        return output


class BottleNeckNormal(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_3_3(int(in_dim / projectionFactor), int(in_dim / projectionFactor))
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckNormal_Asym(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, dropoutRate):
        super(BottleNeckNormal_Asym, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Main branch

        # Secondary branch
        self.block0 = conv_block_1(in_dim, int(in_dim / projectionFactor))
        self.block1 = conv_block_Asym(int(in_dim / projectionFactor), int(in_dim / projectionFactor), 5)
        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block_1(in_dim, out_dim)

    def forward(self, input):
        # Main branch
        # Secondary branch
        b0 = self.block0(input)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        if self.in_dim > self.out_dim:
            output = self.conv_out(input) + do
        else:
            output = input + do
        output = self.PReLU_out(output)

        return output


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, projectionFactor, out_dim):
        super(BottleNeckUpSampling, self).__init__()
        # Main branch
        self.conv0 = nn.Conv2d(in_dim, int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU0 = nn.PReLU()

        self.conv1 = nn.Conv2d(int(in_dim / projectionFactor), int(in_dim / projectionFactor), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(int(in_dim / projectionFactor))
        self.PReLU1 = nn.PReLU()

        self.block2 = conv_block_1(int(in_dim / projectionFactor), out_dim)

        self.do = nn.Dropout(p=0.01)
        self.PReLU3 = nn.PReLU()

    def forward(self, input):
        # Secondary branch
        c0 = self.conv0(input)
        b0 = self.bn0(c0)
        p0 = self.PReLU0(b0)

        c1 = self.conv1(p0)
        b1 = self.bn1(c1)
        p1 = self.PReLU1(b1)

        p2 = self.block2(p1)

        do = self.do(p2)

        return do


class ENet(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.projecting_factor = 4
        self.n_kernels = 16

        # Initial
        self.conv0 = nn.Conv2d(nin, 15, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

        # First group
        self.bottleNeck1_0 = BottleNeckDownSampling(self.n_kernels, self.projecting_factor, self.n_kernels * 4)
        self.bottleNeck1_1 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_2 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_3 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)
        self.bottleNeck1_4 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor, 0.01)

        # Second group
        self.bottleNeck2_0 = BottleNeckDownSampling(self.n_kernels * 4, self.projecting_factor, self.n_kernels * 8)
        self.bottleNeck2_1 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck2_2 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 2)
        self.bottleNeck2_3 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck2_4 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 4)
        self.bottleNeck2_5 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck2_6 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 8)
        self.bottleNeck2_7 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck2_8 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 16)

        # Third group
        self.bottleNeck3_1 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck3_2 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 2)
        self.bottleNeck3_3 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck3_4 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 4)
        self.bottleNeck3_5 = BottleNeckNormal(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor, 0.1)
        self.bottleNeck3_6 = BottleNeckDownSamplingDilatedConv(self.n_kernels * 8, self.projecting_factor,
                                                               self.n_kernels * 8, 8)
        self.bottleNeck3_7 = BottleNeckNormal_Asym(self.n_kernels * 8, self.n_kernels * 8, self.projecting_factor,
                                                   0.1)
        self.bottleNeck3_8 = BottleNeckDownSamplingDilatedConvLast(self.n_kernels * 8, self.projecting_factor,
                                                                   self.n_kernels * 4, 16)

        # ### Decoding path ####
        # Unpooling 1
        self.unpool_0 = nn.MaxUnpool2d(2)

        self.bottleNeck_Up_1_0 = BottleNeckUpSampling(self.n_kernels * 8, self.projecting_factor,
                                                      self.n_kernels * 4)
        self.PReLU_Up_1 = nn.PReLU()

        self.bottleNeck_Up_1_1 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels * 4, self.projecting_factor,
                                                  0.1)
        self.bottleNeck_Up_1_2 = BottleNeckNormal(self.n_kernels * 4, self.n_kernels, self.projecting_factor, 0.1)

        # Unpooling 2
        self.unpool_1 = nn.MaxUnpool2d(2)
        self.bottleNeck_Up_2_1 = BottleNeckUpSampling(self.n_kernels * 2, self.projecting_factor, self.n_kernels)
        self.bottleNeck_Up_2_2 = BottleNeckNormal(self.n_kernels, self.n_kernels, self.projecting_factor, 0.1)
        self.PReLU_Up_2 = nn.PReLU()

        # Unpooling Last
        self.deconv3 = upSampleConv(self.n_kernels, self.n_kernels)

        self.out_025 = nn.Conv2d(self.n_kernels * 8, nout, kernel_size=3, stride=1, padding=1)
        self.out_05 = nn.Conv2d(self.n_kernels, nout, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(self.n_kernels, nout, kernel_size=1)

    def forward(self, input):
        conv_0 = self.conv0(input)  # This will go as res in deconv path
        maxpool_0, indices_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # First group
        bn1_0, indices_1 = self.bottleNeck1_0(outputInitial)
        bn1_1 = self.bottleNeck1_1(bn1_0)
        bn1_2 = self.bottleNeck1_2(bn1_1)
        bn1_3 = self.bottleNeck1_3(bn1_2)
        bn1_4 = self.bottleNeck1_4(bn1_3)

        # Second group
        bn2_0, indices_2 = self.bottleNeck2_0(bn1_4)
        bn2_1 = self.bottleNeck2_1(bn2_0)
        bn2_2 = self.bottleNeck2_2(bn2_1)
        bn2_3 = self.bottleNeck2_3(bn2_2)
        bn2_4 = self.bottleNeck2_4(bn2_3)
        bn2_5 = self.bottleNeck2_5(bn2_4)
        bn2_6 = self.bottleNeck2_6(bn2_5)
        bn2_7 = self.bottleNeck2_7(bn2_6)
        bn2_8 = self.bottleNeck2_8(bn2_7)

        # Third group
        bn3_1 = self.bottleNeck3_1(bn2_8)
        bn3_2 = self.bottleNeck3_2(bn3_1)
        bn3_3 = self.bottleNeck3_3(bn3_2)
        bn3_4 = self.bottleNeck3_4(bn3_3)
        bn3_5 = self.bottleNeck3_5(bn3_4)
        bn3_6 = self.bottleNeck3_6(bn3_5)
        bn3_7 = self.bottleNeck3_7(bn3_6)
        bn3_8 = self.bottleNeck3_8(bn3_7)

        # #### Deconvolution Path ####
        #  First block #
        unpool_0 = self.unpool_0(bn3_8, indices_2)

        # bn_up_1_0 = self.bottleNeck_Up_1_0(unpool_0) # Not concatenate
        bn_up_1_0 = self.bottleNeck_Up_1_0(torch.cat((unpool_0, bn1_4), dim=1))  # concatenate

        up_block_1 = self.PReLU_Up_1(unpool_0 + bn_up_1_0)

        bn_up_1_1 = self.bottleNeck_Up_1_1(up_block_1)
        bn_up_1_2 = self.bottleNeck_Up_1_2(bn_up_1_1)

        #  Second block #
        unpool_1 = self.unpool_1(bn_up_1_2, indices_1)

        # bn_up_2_1 = self.bottleNeck_Up_2_1(unpool_1) # Not concatenate
        bn_up_2_1 = self.bottleNeck_Up_2_1(torch.cat((unpool_1, outputInitial), dim=1))  # concatenate

        bn_up_2_2 = self.bottleNeck_Up_2_2(bn_up_2_1)

        up_block_1 = self.PReLU_Up_2(unpool_1 + bn_up_2_2)

        unpool_12 = self.deconv3(up_block_1)

        return self.final(unpool_12)


class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3
    


class ResidualCoordConvUNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32):
        super().__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        # Encoder
        self.down_1 = CoordConv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = CoordConv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = CoordConv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = CoordConv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = CoordConv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # Decoder
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = CoordConv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = CoordConv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = CoordConv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_decod_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = CoordConv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # Params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        print(f"Initialized {self.__class__.__name__} succesfully")

    def forward(self, input):
        # Encoding path

        down_1 = self.down_1(input)  # This will go as res in deconv path
        down_2 = self.down_2(self.pool_1(down_1))
        down_3 = self.down_3(self.pool_2(down_2))
        down_4 = self.down_4(self.pool_3(down_3))

        bridge = self.bridge(self.pool_4(down_4))

        # Decoding path
        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2  # Residual connection
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2  # Residual connection
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2  # Residual connection
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2  # Residual connection
        up_4 = self.up_4(skip_4)

        return self.out(up_4)



    
    
    
    
    
class ResidualUNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32):
        super().__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        # Encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # Bridge between Encoder-Decoder
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # Decoder
        self.deconv_1 = conv_decod_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_decod_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # Params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        print(f"Initialized {self.__class__.__name__} succesfully")

    def forward(self, input):
        # Encoding path

        down_1 = self.down_1(input)  # This will go as res in deconv path
        down_2 = self.down_2(self.pool_1(down_1))
        down_3 = self.down_3(self.pool_2(down_2))
        down_4 = self.down_4(self.pool_3(down_3))

        bridge = self.bridge(self.pool_4(down_4))

        # Decoding path
        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2  # Residual connection
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2  # Residual connection
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2  # Residual connection
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2  # Residual connection
        up_4 = self.up_4(skip_4)

        return self.out(up_4)


class BBConv(Module):
    def __init__(self, in_feat, out_feat, pool_ratio):
        super(BBConv, self).__init__()
        self.mp = nn.MaxPool2d(pool_ratio)
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = F.sigmoid(x)
        return x
    
    
    
class BB_Unet(Module):
    """A reference U-Net model.
    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1, no_grad=False, n_organs = 4):
        super(BB_Unet, self).__init__()
        if no_grad is True:
            no_grad_state = True
        else:
            no_grad_state = False
        
        #Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottle neck
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)
        # bounding box encoder path:
        self.b1 = BBConv(n_organs, 256, 4)
        self.b2 = BBConv(n_organs, 128, 2)
        self.b3 = BBConv(n_organs, 64, 1)
        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, n_organs, kernel_size=3, padding=1)

    def forward(self, x, bb, mode= 'Tr'):
        x1 = self.conv1(x)
        p1 = self.mp1(x1)

        x2 = self.conv2(p1)
        p2 = self.mp2(x2)

        x3 = self.conv3(p2)
        p3 = self.mp3(x3)

        # Bottle neck
        x4 = self.conv4(p3)
        if mode == 'tr':
            f1_1 = self.b1(bb)
            f2_1 = self.b2(bb)
            f3_1 = self.b3(bb)
            x3_1 = x3*f1_1
            x2_1 = x2*f2_1
            x1_1 = x1*f3_1
        elif mode == 'ev':
            x3_1 = x3
            x2_1 = x2
            x1_1 = x1
        u1 = self.up1(x4, x3_1)
        u2 = self.up2(u1, x2_1)
        u3 = self.up3(u2, x1_1)

        x5 = self.conv9(u3)

        return x5


