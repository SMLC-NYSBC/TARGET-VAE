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
    def __init__(self, n_in, n_out, activation=nn.Tanh):
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
    
    

class InferenceNetwork(nn.Module):
    def __init__(self, n, latent_dim, hidden_dim, num_layers=1, activation=nn.Tanh, resid=False):
        super(InferenceNetwork, self).__init__()

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


class SpatialGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_out=1, num_layers=1, activation=nn.Tanh
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
    def __init__(self, n, latent_dim, hidden_dim, n_out=1, num_layers=1, activation=nn.Tanh
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



    

class MNIST_ConvInferenceNetwork(nn.Module):
    def __init__(self, n, latent_dim, activation=nn.Tanh, resid=False):
        super(MNIST_ConvInferenceNetwork, self).__init__()

        self.latent_dim = latent_dim
        self.n = n
        
        self.activation = activation()
        
        self.conv1 = nn.Conv2d(1, 32, 5, padding=3)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.conv3 = nn.Conv2d(64, 128, 1)
        
        self.conv4 = nn.Conv2d(128, 2*latent_dim, 28)
        
        
        

    def forward(self, x):
        #x = x.view(-1, 1, 28, 28)
        
        x = self.activation(self.conv1(x))
        #print(x.shape)
        
        x = self.activation(self.conv2(x))
        #print(x.shape)
        
        x = self.activation(self.conv3(x))
        #print(x.shape)
        
        z = self.conv4(x)
        #print(z.shape)
        
        ld = self.latent_dim
        z = z.view(-1, 2*self.latent_dim)
        
        z_mu = z[:,:ld]
        z_logstd = z[:,ld:]

        return z_mu, z_logstd
  

    
class Particles_ConvInferenceNetwork(nn.Module):
    def __init__(self, n, latent_dim, activation=nn.Tanh, resid=False):
        super(HDB_ConvInferenceNetwork, self).__init__()

        self.latent_dim = latent_dim
        self.n = n
        
        self.activation = activation()
        
        self.conv1 = nn.Conv2d(1, 32, 5, padding=3)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.conv3 = nn.Conv2d(64, 128, 1)
        
        self.conv4 = nn.Conv2d(128, 2*latent_dim, 40)
       
        

    def forward(self, x):
        #x = x.view(-1, 1, 40, 40)
        
        x = self.activation(self.conv1(x))
        #print(x.shape)
        
        x = self.activation(self.conv2(x))
        #print(x.shape)
        
        x = self.activation(self.conv3(x))
        #print(x.shape)
        
        z = self.conv4(x)
        #print(z.shape)
        
        ld = self.latent_dim
        z = z.view(-1, 2*self.latent_dim)
        
        z_mu = z[:,:ld]
        z_logstd = z[:,ld:]

        return x, z_mu, z_logstd



class MNIST_AttentionNetwork(nn.Module):
    
    def __init__(self, latent_dim=2, activation=nn.Tanh):
        
        super(MNIST_AttentionNetwork, self).__init__()
        
        self.activation = activation()
        self.latent_dim = latent_dim
        
	self.conv1 = nn.Conv2d(1, 500, 28, padding=27)
        self.conv2 = nn.Conv2d(500, 500, 1)

        self.conv_a = nn.Conv2d(500, 1, 1)
        self.conv_r = nn.Conv2d(500, 2, 1)
        
        self.conv_z = nn.Conv2d(500, 2*self.latent_dim, 1)
        
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        
        x = self.activation(self.conv1(x))
        h = self.activation(self.conv2(x))
        
        a = self.conv_a(h)
        a = a.view(a.shape[0], -1)
        
        p = F.gumbel_softmax(a)
        #p = F.softmax(a, dim=-1)
        
        z = self.conv_z(h)
        
        theta = self.conv_r(h)
        
        return p, theta, z

    
    
    
    
class GroupConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
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
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
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
        
        #rotate self.weight as much as output_stabilizer_size, each time by 2*pi/output_stabilizer_size,
        #and then concatentate the values
        
        res = torch.zeros(self.weight.shape[0], self.output_stabilizer_size, 
                          self.weight.shape[1], self.weight.shape[2],
                          self.weight.shape[3], self.weight.shape[4]).cuda()
        d_theta = 2*np.pi / self.output_stabilizer_size
        theta = 0.0
        
        for i in range(self.output_stabilizer_size):
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
        
        tw_shape = (self.out_channels*self.output_stabilizer_size,
                    self.in_channels*self.input_stabilizer_size,
                    self.ksize, self.ksize)
        
        tw = tw.view(tw_shape)
        
        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], 
                           input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y
    



    

    

    
class MNIST_AttentionNetwork_groupconv(nn.Module):
    
    def __init__(self, latent_dim=2, activation=nn.LeakyReLU):
        
        super(MNIST_AttentionNetwork_groupconv_myGroupConv_2, self).__init__()
        
        self.activation = activation()
        self.latent_dim = latent_dim
        
        self.conv1 = GroupConv(1, 128, 28, padding=27, input_stabilizer_size=1, output_stabilizer_size=4)
        self.conv2 = nn.Conv3d(128, 128, 1)
        
        self.conv_a = nn.Conv3d(128, 1, 1)
        self.conv_r = nn.Conv3d(128, 2, 1)
        self.conv_z = nn.Conv3d(128, 2*self.latent_dim, 1)
        
        
        
    def forward(self, x, epoch):
        x = x.view(-1, 1, 1, 28, 28)
        
        x = self.activation(self.conv1(x))
        h = self.activation(self.conv2(x))
        
        # calculate rotation from group conv features; attn_values for rotations at each patch; which rotation at each patch!
        attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)
        a = attn.view(attn.shape[0], -1)
        
        p = F.gumbel_softmax(a, dim=-1, tau=0.1 )
        p = p.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])
        
        
        # calculate rotation from group conv features
        angles1 = torch.tensor([0, np.pi/2, np.pi, 3*np.pi/2]).type(torch.float).cuda()
        angles2 = torch.ones_like(p) * angles1.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        theta_raw = self.conv_r(h) 
        
        #sgn = torch.sign(theta_raw[:,0, 2,:,:]).detach()
        #angles2[:,2, : , :] = angles2[:,2, : , :] * sgn
        
        theta_mu = theta_raw[:,0,:,:,:] + angles2
        theta_std = theta_raw[:, 1,:,:,:] 
        theta = torch.stack((theta_mu, theta_std), dim=1)
        
        z = self.conv_z(h)
        
        
        return attn, p, theta, z     
    
    

class MNIST_AttentionNetwork_groupconv_myGroupConv_NoSampling(nn.Module):
    
    def __init__(self, latent_dim=2, activation=nn.LeakyReLU):
        
        super(MNIST_AttentionNetwork_groupconv_myGroupConv_NoSampling, self).__init__()
        
        self.activation = activation()
        self.latent_dim = latent_dim
        
        self.conv1 = GroupConv(1, 128, 28, padding=27, input_stabilizer_size=1, output_stabilizer_size=4)
        self.conv2 = GroupConv(128, 128, 1, input_stabilizer_size=4, output_stabilizer_size=4)
        
        
        #self.rearrange = Rearrange('b c r h w -> b c r (h w)')
        #self.avg_pooling = Reduce('b c r h w -> b c 1 h w', 'mean')
        
        self.conv_a = nn.Conv3d(128, 1, 1)
        
        self.conv_z = nn.Conv3d(128, 2*self.latent_dim, 1)
        
        
    def forward(self, x):
        x = x.view(-1, 1, 1, 28, 28)
        
        x = self.activation(self.conv1(x))
        h = self.activation(self.conv2(x))
        
        # calculate attn_values for rotations at each patch
        a = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)
        a = a.view(a.shape[0], -1)
        p = F.gumbel_softmax(a, dim=-1)
        #p = F.softmax(a , dim=-1)
        p = p.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4]).type(torch.float)
        
        # calculate rotation from group conv features
        angles = torch.tensor(np.arange(0, 2*np.pi, np.pi/2)).type(torch.float).cuda()
        angles = angles.expand(p.shape[0], angles.shape[0]).unsqueeze(2).unsqueeze(3)
        angles = torch.zeros_like(p) + angles
        
        z = self.conv_z(h)
        
        return p, angles, z 
    
    
    
