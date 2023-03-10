#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 06:00:09 2019

@author: aneesh
"""

import torch
import torch.nn as nn



class stride_supression(nn.Module):
    def __init__(self, stride, dimension=1):
        super(stride_supression, self).__init__()
        self.d = dimension
        self.stride = stride
    
    def create_subcubes(self, stride, layer):
        if isinstance(stride, int) == True:
            stride = (stride, stride, stride)
        
        result = []
        for z in range(stride[2]):
            for y in range(stride[1]):
                for x in range(stride[0]):
                    result.append(layer[..., x::stride[0], y::stride[1], z::stride[2]])

        return result

    def forward(self, x):
        return torch.cat(self.create_subcubes(self.stride, x), 1)



class ResnetGenerator3DSSP(nn.Module):
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
        super(ResnetGenerator3D, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        
        stem = [nn.Conv3d(1, ngf, kernel_size=(1, 1, 6), padding=(0, 0, 3), stride=1),
                 stride_supression((2, 2, 1)),
                 norm_layer(ngf * 2 * 2, affine=True),
                 nn.ReLU(True),
                 nn.Conv3d(ngf * 2 * 2, ngf * 2, kernel_size=1, padding=0),
                 norm_layer(ngf * 2)]
                

        down = []
        n_downsampling = 1
        for i in range(n_downsampling):
            mult = 2**n_downsampling
            down +=  [ResnetBlock(ngf * mult * 2, 'zero', input_dim = ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout, 
                              stride = 2, kernel=3)]

        middle = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            middle += [ResnetBlock(ngf * mult * 2, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        up = []
        
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            up += [nn.ConvTranspose3d(ngf * mult, ngf * mult // 2,
                                         kernel_size=3, stride=(2, 2, 1),
                                         padding=1, output_padding=(1, 1, 0)),
                      norm_layer(ngf * mult // 2, affine=True),
                      nn.ReLU(True)]
            

        self.final = nn.Conv3d(ngf, output_nc, kernel_size=(7,7,26), padding=(3,3,0))
        
        self.stem   = nn.Sequential(*stem)
        self.down   = nn.Sequential(*down)
        self.middle = nn.Sequential(*middle)
        self.up     = nn.Sequential(*up)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            x = self.stem(input)
            x = self.down(x)
            x = self.middle(x)
            x = self.up(x)
            x = self.final(x)
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
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, input_dim = None, use_bias = False, stride = 1, kernel = 3):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, input_dim, use_bias, stride, kernel)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, input_dim, use_bias, stride, kernel):
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

        input_dim = dim if input_dim == None else input_dim
        if stride == 1:
            conv_block += [nn.Conv3d(input_dim, dim, kernel_size=kernel, padding=p, bias=use_bias, stride = stride),
                        norm_layer(dim),
                        nn.ReLU(True)]
            dimF = dim
        else:
            dimF = dim * (stride**3)   
            conv_block += [nn.Conv3d(input_dim, dim, kernel_size=kernel, padding=p, bias=use_bias, stride = 1),
                           stride_supression(stride),
                        norm_layer(dimF),
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
        
        
        conv_block += [nn.Conv3d(dimF, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_dim, dim, kernel_size=1, stride=stride, bias=use_bias),
                norm_layer(dim)
            )
        else:
            self.shortcut = nn.Sequential(*[])

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.shortcut(x) + self.conv_block(x))
        return out
