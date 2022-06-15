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
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn.init as init

import math
import numpy as np



class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.LeakyReLU):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)

    
class RandomFourierEmbedding2d(nn.Module):
    def __init__(self, in_dim, embedding_dim, sigma=0.01, learnable=False):
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
                , softplus=False, resid=False, fourier_expansion=False, sigma=0.01):
        super(SpatialGenerator, self).__init__()

        self.softplus = softplus
        self.fourier_expansion = fourier_expansion

        in_dim = 2
        if fourier_expansion:
            embedding_dim = 1024
            self.embed_latent = RandomFourierEmbedding2d(in_dim, embedding_dim, sigma, learnable=True).cuda()
            in_dim = embedding_dim
            print('# in_dim is {}'.format(in_dim))
            

        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

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
        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        
        x = x.view(b*n, -1)
        
        if self.fourier_expansion:
            x = self.embed_latent(x)
            
        
        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        h = h_x + h_z # (batch, num_coords, hidden_dim)
        h = h.view(b*n, -1)

        y = self.layers(h) # (batch*num_coords, nout)
        y = y.view(b, n, -1)

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
            out_channels, in_channels, self.input_rot_dim, *kernel_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
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
    '''
    Inference without attention on the translation and rotation values
    '''
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
        z = self.layers(x)

        ld = self.latent_dim
        z_mu = z[:,:ld]
        z_logstd = z[:,ld:]

        return z_mu, z_logstd






    
class InferenceNetwork_AttentionTranslation_UnimodalRotation(nn.Module):
    '''
    Inference with attention only on the translation values
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, activation=nn.LeakyReLU, groupconv=0):

        super(InferenceNetwork_AttentionTranslation_UnimodalRotation, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.groupconv = groupconv

        if self.groupconv == 0:
            self.conv1 = nn.Conv2d(in_channels, self.kernels_num, self.input_size, padding=self.input_size//2)
            self.conv2 = nn.Conv2d(self.kernels_num, self.kernels_num, 1)
            
            self.conv_a = nn.Conv2d(self.kernels_num, 1, 1)
            self.conv_r = nn.Conv2d(self.kernels_num, 2, 1)
            self.conv_z = nn.Conv2d(self.kernels_num, 2*self.latent_dim, 1)
        else:
            self.conv1 = GroupConv(1, self.kernels_num, self.input_size, padding=self.input_size//2, input_rot_dim=1, output_rot_dim=self.groupconv)
            self.conv2 = nn.Conv2d(self.kernels_num*self.groupconv, self.kernels_num*self.groupconv, 1)
            self.fc_r = nn.Linear(self.groupconv, 1)
            
            self.conv_a = nn.Conv2d(self.kernels_num*self.groupconv, 1, 1)
            self.conv_r = nn.Conv2d(self.kernels_num*self.groupconv, 2, 1)
            self.conv_z = nn.Conv2d(self.kernels_num*self.groupconv, 2*self.latent_dim, 1)
            
             
        
    def forward(self, x):
        x = self.activation(self.conv1(x))

        if self.groupconv > 0:
            x = x.permute(0, 1, 3, 4, 2)
            x = self.fc_r(x).squeeze(4)
            #x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])

        h = self.activation(self.conv2(x))    

        attn = self.conv_a(h)
        a = attn.view(attn.shape[0], -1)
        a_sampled = F.gumbel_softmax(a, dim=-1)
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3])

        z = self.conv_z(h)

        theta = self.conv_r(h)

        return attn, a_sampled, theta, z

 
    
    


class InferenceNetwork_AttentionTranslation_AttentionRotation(nn.Module):
    '''
    Inference with attention on both the translation and rotation values (inference model for TARGET-VAE)
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, kernels_size=65, padding=16, activation=nn.LeakyReLU
                 , groupconv=0, rot_refinement=False, theta_prior=np.pi, normal_prior_over_r=True):

        super(InferenceNetwork_AttentionTranslation_AttentionRotation, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.kernels_size = kernels_size
        self.padding = padding
        self.groupconv = groupconv
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        print('self.normal_prior_over_r is {}'.format(self.normal_prior_over_r) )

        self.conv1 = GroupConv(in_channels, self.kernels_num, self.kernels_size, padding=self.padding, input_rot_dim=1, output_rot_dim=self.groupconv)
        self.conv2 = nn.Conv3d(self.kernels_num, self.kernels_num, 1)

        self.conv_a = nn.Conv3d(self.kernels_num, 1, 1)
        self.conv_r = nn.Conv3d(self.kernels_num, 2, 1)
        self.conv_z = nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        h = self.activation(self.conv2(x)) 

        attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)
        
        if self.rot_refinement:
            if self.groupconv == 4:
                offsets = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]).type(torch.float).cuda() 
            elif self.groupconv == 8:
                offsets = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]).type(torch.float).cuda()
            elif self.groupconv == 16:
                offsets = torch.tensor([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi, -7*np.pi/8, -3*np.pi/4, -5*np.pi/8, -np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8]).type(torch.float).cuda()
            
            if self.normal_prior_over_r:
                prior_theta = Normal(torch.tensor([0.0]).cuda(), torch.tensor([self.theta_prior]).cuda())
            else:
                prior_theta = Uniform(torch.tensor([-2*np.pi]).cuda(), torch.tensor([2*np.pi]).cuda())
            p_r = prior_theta.log_prob(offsets).unsqueeze(1).unsqueeze(2)
                
        else:
            # uniform prior over r when no offsets are being added to the rot_means
            p_r = torch.zeros(self.groupconv).cuda() - np.log(attn.shape[1])
            p_r = p_r.unsqueeze(1).unsqueeze(2)
            
        
        attn = attn + p_r
        q_t_r = F.log_softmax(attn.view(attn.shape[0], -1), dim=1).view(attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W
        
        a = attn.view(attn.shape[0], -1)
        
        a_sampled = F.gumbel_softmax(a, dim=-1) #
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])

        z = self.conv_z(h)

        theta = self.conv_r(h)
        
        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ] 
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).cuda()
            
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z



    
    
