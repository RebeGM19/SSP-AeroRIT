#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 06:00:09 2019

@author: aneesh
"""

import torch
import torch.nn as nn

class ResnetGenerator3DPre(nn.Module):
    '''
    Construct a Resnet-based generator
    Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Parameters:
        input_nc (int)      -- the number of channels in input images
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        gpu_ids             -- GPUs for parallel processing
    '''
    def __init__(self, input_nc, output_nc, ngf=16, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator3DPre, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        #stem = [nn.Conv3d(1, ngf, kernel_size=(7, 7, 6), padding=3),
                 #norm_layer(ngf, affine=True),
                 #nn.ReLU(True)]
                 
        stem = [nn.Conv3d(1, ngf, kernel_size=(1, 1, 6), padding=(0, 0, 3), stride=(2, 2, 1)),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True),
                 nn.Conv3d(ngf, ngf * 2, kernel_size=(1, 1, 1)),
                 norm_layer(ngf * 2)]       # En el ALTER

        down = []
        n_downsampling = 1  # En el ALTER
        #n_downsampling = 2
        for i in range(n_downsampling):
            #mult = 2**i
            mult = 2**n_downsampling  # En el ALTER
            down += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        middle = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            middle += [ResnetBlock(ngf * mult * 2, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        up = []
        
        n_downsampling = 2  # En el ALTER
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            up += [nn.ConvTranspose3d(ngf * mult, ngf * mult // 2,
                                         kernel_size=3, stride=(2, 2, 1),
                                         padding=1, output_padding=(1, 1, 0)),
                      norm_layer(ngf * mult // 2, affine=True),
                      nn.ReLU(True)]
                      
        self.final = nn.Conv3d(ngf, output_nc, kernel_size=(7,7,26), padding=(3,3,0))#, stride=(1,1,7))
        #self.final = nn.Conv3d(ngf, output_nc, kernel_size=(1,1,26), padding=0, stride=(2, 2, 1))#, stride=(1,1,7))
        
        
        #self.final = nn.Conv3d(ngf, output_nc, kernel_size=(7,7,26), padding=(3,3,0))


        self.stem   = nn.Sequential(*stem)
        self.down   = nn.Sequential(*down)
        self.middle = nn.Sequential(*middle)
        self.up     = nn.Sequential(*up)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            #print(input.shape)
            x = self.stem(input)
            #print(x.shape)
            x = self.down(x)
            #print(x.shape)
            x = self.middle(x)
            #print(x.shape)
            x = self.up(x)
            #print(x.shape)
            x = self.final(x)
            #print(x.shape)
            #exit()
            return x

class ResnetBlock(nn.Module):
    '''
    Defines a ResNet block
    Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Parameters:
        dim (int)           -- the number of channels in the conv layer.
        padding_type (str)  -- the name of padding layer: reflect | replicate | zero
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
        use_bias (bool)     -- if the conv layer uses bias or not
    '''
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias = False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        #print(x.shape)
        #print(self.conv_block(x).shape)
        out = x + self.conv_block(x)
        return out