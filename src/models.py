from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Parameter
from torch.autograd import Variable
import torch.utils.data

import math
import numpy as np

from einops.layers.torch import Rearrange, Reduce


class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.LeakyReLU):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)

    
class RandomFourierEmbedding2d(nn.Module):
    def __init__(self, in_dim, embedding_dim, sigma=0.2, learnable=False):
        super(RandomFourierEmbedding2d, self).__init__()

        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        
        #self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

        w = torch.randn(embedding_dim, in_dim) #/ self.sigma  shape of weights: (out_features, in_features)
        b = torch.rand(embedding_dim)*2*np.pi

        if learnable:
            self.weight = nn.Parameter(w)
            self.bias = nn.Parameter(b)
            
        else:
            self.register_buffer('weight', w)
            self.register_buffer('bias', b)

        print('# sigma value is {}'.format(self.sigma))
        
        
        

    def forward(self, x):
        if x is None:
            return 0
        #z = np.sqrt(2) * torch.cos(F.linear(x, self.weight, self.bias)) / np.sqrt(self.embedding_dim)
        
        z = torch.cos(F.linear(x, self.weight/self.sigma, self.bias)) 
        
        return z
    
    




class SpatialGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_out=1, num_layers=1, activation=nn.LeakyReLU
                , softplus=False, resid=False, expand_coords=False, bilinear=False):
        super(SpatialGenerator, self).__init__()

        self.softplus = softplus
        self.expand_coords = expand_coords

        in_dim = 2
        if expand_coords:
            embedding_dim = 1024
            self.embed_latent = RandomFourierEmbedding2d(in_dim, embedding_dim, learnable=True).cuda()
            in_dim = embedding_dim
            print('# in_dim is {}'.format(in_dim))
            

        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

        if latent_dim > 0 and bilinear: # include bilinear layer on latent and coordinates
            self.bilinear = nn.Bilinear(in_dim, latent_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1,num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)

        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        
        x = x.view(b*n, -1)
        
        if self.expand_coords:
            x = self.embed_latent(x)
        
        
        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        h_bi = 0
        if hasattr(self, 'bilinear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            z = z.unsqueeze(1) # broadcast over coordinates
            x = x.view(b, n, -1)
            z = z.expand(b, x.size(1), z.size(2)).contiguous()
            h_bi = self.bilinear(x, z)

        h = h_x + h_z + h_bi # (batch, num_coords, hidden_dim)
        h = h.view(b*n, -1)

        y = self.layers(h) # (batch*num_coords, nout)
        y = y.view(b, n, -1)

        if self.softplus: # only apply softplus to first output
            y = torch.cat([F.softplus(y[:,:,:1]), y[:,:,1:]], 2)

        return y


class VanillaGenerator(nn.Module):
    def __init__(self, n, latent_dim, hidden_dim, n_out=1, num_layers=1, activation=nn.LeakyReLU
                , softplus=False, resid=False):
        super(VanillaGenerator, self).__init__()
        """
        The standard MLP structure for image generation. Decodes each pixel location as a funciton of z.
        """

        self.n_out = n_out
        self.softplus = softplus

        layers = [nn.Linear(latent_dim,hidden_dim), 
                  activation()]
        for _ in range(1,num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n*n_out))
        if softplus:
            layers.append(nn.Softplus())

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)

        # ignores x, decodes each pixel conditioned on z

        y = self.layers(z).view(z.size(0), -1, self.n_out)
        if self.softplus: # only apply softplus to first output
            y = torch.cat([F.softplus(y[:,:,:1]), y[:,:,1:]], 2)

        return y



    
    
    
    
    
class GroupConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_rot_dim=1, output_rot_dim=4):
        super(GroupConv, self).__init__()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_rot_dim = input_rot_dim
        self.output_rot_dim = output_rot_dim

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_rot_dim, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        
        
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    
    
    def trans_filter(self):
        
        #rotate self.weight as much as output_rot_dim, each time by 2*pi/output_rot_dim,
        #and then concatentate the values
        
        res = torch.zeros(self.weight.shape[0], self.output_rot_dim, 
                          self.weight.shape[1], self.weight.shape[2],
                          self.weight.shape[3], self.weight.shape[4]).cuda()
        d_theta = 2*np.pi / self.output_rot_dim
        theta = 0.0
        
        for i in range(self.output_rot_dim):
            #create the rotation matrix
            rot = torch.zeros(self.weight.shape[0], 3, 4).cuda()
            rot[:,0,0] = np.cos(theta)
            rot[:,0,1] = np.sin(theta)
            rot[:,1,0] = -np.sin(theta)
            rot[:,1,1] = np.cos(theta)


            
            grid = F.affine_grid(rot, self.weight.shape, align_corners=False)
            res[:, i, :, :, :] = F.grid_sample(self.weight, grid, align_corners=False)
            
            theta += d_theta
        
        return res
    
    
        

    def forward(self, input):
        tw = self.trans_filter()
        
        tw_shape = (self.out_channels*self.output_rot_dim,
                    self.in_channels*self.input_rot_dim,
                    self.ksize, self.ksize)
        
        tw = tw.view(tw_shape)
        
        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_rot_dim, input_shape[-2], 
                           input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_rot_dim, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y
    


class InferenceNetwork_UnimodalTranslation_UnimodalRotation(nn.Module):
    def __init__(self, n, latent_dim, hidden_dim, num_layers=1, activation=nn.LeakyReLU, resid=False):
        super(InferenceNetwork_UnimodalTranslation_UnimodalRotation, self).__init__()

        self.latent_dim = latent_dim
        self.n = n

        layers = [nn.Linear(n, hidden_dim),
                  activation(),
                 ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        layers.append(nn.Linear(hidden_dim, 2*latent_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x is (batch,num_coords)
        z = self.layers(x)

        ld = self.latent_dim
        z_mu = z[:,:ld]
        z_logstd = z[:,ld:]

        return z_mu, z_logstd






    
class InferenceNetwork_AttentionTranslation_UnimodalRotation(nn.Module):
    
	def __init__(self, n, latent_dim, kernels_num=128, activation=nn.LeakyReLU, groupconv=groupconv):
        
		super(InferenceNetwork_AttentionTranslation_UnimodalRotation, self).__init__()
        
		self.activation = activation()
		self.latent_dim = latent_dim
		self.input_size = n
		self.kernels_num = kernels_num
		self.translaion_inference = translaion_inference
		self.rotation_inference = rotation_inference
		self.groupconv = groupconv
		
		if self.groupconv == 0:
			self.conv1 = nn.Conv2d(1, self.kernels_num, self.input_size, padding=self.input_size-1)
			self.conv2 = nn.Conv2d(self.kernels_num, self.kernels_num, 1)
		else:
			self.conv1 = GroupConv(1, self.kernels_num, self.input_size, padding=self.input_size-1, input_rot_dim=1, output_rot_dim=self.groupconv)
			self.conv2 = nn.Conv3d(self.kernels_num, self.kernels_num, 1)
			self.avg_pooling_layer = Reduce('b c r h w -> b c 1 h w', 'mean')

		self.conv_a = nn.Conv2d(self.kernels_num, 1, 1)
		self.conv_r = nn.Conv2d(self.kernels_num, 2, 1)
		self.conv_z = nn.Conv2d(self.kernels_num, 2*self.latent_dim, 1)
        
        
	def forward(self, x, epoch):
		x = x.view(-1, 1, self.input_size, self.input_size)
        
		x = self.activation(self.conv1(x))
		h = self.activation(self.conv2(x))
		
		if self.groupconv > 0:
			h = self.avg_pooling_layer(h).squeeze(2)

		attn = self.conv_a(h)
		a = attn.view(attn.shape[0], -1)
		p = F.gumbel_softmax(a, dim=-1)
		p = p.view(h.shape[0], h.shape[2], h.shape[3])

		z = self.conv_z(h)

		theta = self.conv_r(h)

		return attn, p, theta, z





class InferenceNetwork_AttentionTranslation_UnimodalRotation_GroupConv(nn.Module):
    
	def __init__(self, n, latent_dim, kernels_num=128, activation=nn.LeakyReLU, translation_inference='unimodal', rotation_inference='unimodal', groupconv=0):
        
		super(InferenceNetwork_AttentionTranslation_UnimodalRotation_GroupConv, self).__init__()
        
		self.activation = activation()
		self.latent_dim = latent_dim
		self.input_size = n
		self.kernels_num = kernels_num
		self.translaion_inference = translaion_inference
		self.rotation_inference = rotation_inference
		self.groupconv = groupconv

		self.conv1 = GroupConv(1, self.kernels_num, self.input_size, padding=self.input_size-1, input_rot_dim=1, output_rot_dim=self.groupconv)
		self.conv2 = nn.Conv3d(self.kernels_num, self.kernels_num, 1)
		self.conv_a = nn.Conv3d(self.kernels_num, 1, 1)
		self.avg_pooling_layer = Reduce('b c r h w -> b c 1 h w', 'mean')
		self.conv_r = nn.Conv2d(self.kernels_num, 2, 1)
		self.conv_z = nn.Conv2d(self.kernels_num, 2*self.latent_dim, 1)
        	

        
        
	def forward(self, x, epoch):
		x = x.view(-1, 1, self.input_size, self.input_size)
        
		x = self.activation(self.conv1(x))
		h = self.activation(self.conv2(x))
		
		h = self.avg_pooling_layer(h).squeeze(2)		

		# calculate rotation from group conv features; attn_values for rotations at each patch
		attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)
		a = attn.view(attn.shape[0], -1)
		p = F.gumbel_softmax(a, dim=-1)
		p = p.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])
		
		

		z = self.conv_z(h)

		theta = self.conv_r(h)

		return attn, p, theta, z



'''
		x = x.view(-1, 1, 1, self.input_size, self.input_size)

		x = self.activation(self.conv1(x))
		h = self.activation(self.conv2(x))

		# calculate rotation from group conv features; attn_values for rotations at each patch; which rotation at each patch!
		attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)
		a = attn.view(attn.shape[0], -1)

		p = F.gumbel_softmax(a, dim=-1, tau=0.1 )
		p = p.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])


		# calculate rotation from group conv features
		rotation_offset = torch.tensor(np.arange(0, 2*np.pi, 2*np.pi/self.output_rot_dim)).type(torch.float).cuda()
		rotation_offset = torch.ones_like(p) * rotation_offset.unsqueeze(0).unsqueeze(2).unsqueeze(3)
		r_values = self.conv_r(h) 

		theta_mu = r_values[:,0,:,:,:] + rotation_offset
		theta_std = r_values[:, 1,:,:,:] 
		theta = torch.stack((theta_mu, theta_std), dim=1)

		z = self.conv_z(h)


		return attn, p, theta, z  
'''













		   
    
    


