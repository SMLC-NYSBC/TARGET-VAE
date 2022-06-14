from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys
import os
import datetime
import shutil

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim.lr_scheduler import ReduceLROnPlateau

import src.models as models
import src.mrc as mrc
import src.image as image_utils
import src.ctf as C
from src.utils import EarlyStopping


def eval_minibatch(x, y, ctf, generator_model, encoder_model, translation_inference, rotation_inference, epoch, device
                  , theta_prior, groupconv, padding, mask_radius):

    b = y.size(0)
    n = int(y.size(-1))
    btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()
    x = x.expand(b, x.size(0), x.size(1)).to(device)
    y = y.to(device)

    if translation_inference == 'unimodal' and rotation_inference == 'unimodal':
        y = y.view(b, -1)
        z_mu,z_logstd = encoder_model(y)
        z_std = torch.exp(z_logstd)
        z_dim = z_mu.size(1)

        # draw samples from variational posterior to calculate E[p(x|z)]
        r = Variable(x.data.new(b,z_dim).normal_())
        z = z_std*r + z_mu

        kl_div = 0
        # z[0] is the rotation
        theta_mu = z_mu[:,0]
        theta_std = z_std[:,0]
        theta_logstd = z_logstd[:,0]
        theta = z[:,0]
        z = z[:,1:]
        z_mu = z_mu[:,1:]
        z_std = z_std[:,1:]
        z_logstd = z_logstd[:,1:]

        # calculate the KL divergence term
        sigma = theta_prior
        kl_div = -theta_logstd + np.log(sigma) + (theta_std**2 + theta_mu**2)/2/sigma**2 - 0.5

        # z[0,1] are the translations
        dx_scale = 0.1
        dx_mu = z_mu[:,:2]
        dx_std = z_std[:,:2]
        dx_logstd = z_logstd[:,:2]
        dx = z[:,:2]*dx_scale # scale dx by standard deviation
        dx = dx.unsqueeze(1)
        z = z[:,2:]
        
        x = x - dx # translate coordinates
        
        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta
        
        
        # unit normal prior over z and translation
        z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
        kl_div = kl_div + torch.sum(z_kl, 1)
        kl_div = kl_div.mean()


    elif translation_inference == 'attention' and rotation_inference == 'unimodal':
        rand_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        
        attn, attn_sampled, theta_vals, z_vals = encoder_model(y)
        
        #attn_sampled returned here is over the locations since rotation_inference is unimodal
        attn_sampled = attn_sampled.view(attn_sampled.shape[0], -1).unsqueeze(2)
        z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
        theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)


        z_dim = z_vals.size(1) // 2
        z_mu = z_vals[:,:z_dim, ]
        z_logstd = z_vals[:, z_dim:, ]
        z_std = torch.exp(z_logstd)

        z_mu_expected = torch.bmm(z_mu, attn_sampled)
        z_std_expected = torch.bmm(z_std, attn_sampled)

        # draw samples from variational posterior to calculate
        r_z = rand_dist.sample((b, z_dim)).to(device)
        z = (z_std_expected*r_z + z_mu_expected).squeeze(2)

        attn_dim = attn.shape[3]
        if  attn_dim % 2:
            x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)
            y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)[::-1]
        else:
            x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)
            y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)[::-1]
        x_0,x_1 = np.meshgrid(x_grid, y_grid)
        x_translated_sample = np.stack([x_0.ravel(), x_1.ravel()], 1)
        x_translated_sample = torch.from_numpy(x_translated_sample).to(device)
        x_translated_batch = x_translated_sample.expand(b, x_translated_sample.size(0), x_translated_sample.size(1))
        x_translated_batch = x_translated_batch.transpose(1, 2)
        dx = torch.bmm(x_translated_batch.type(torch.float), attn_sampled).squeeze(2).unsqueeze(1)
        x = x - dx # translate coordinates

        eps = 1e-6

        theta_mu = theta_vals[:, 0:1, ]
        theta_logstd = theta_vals[:, 1:2, ] 
        theta_std = torch.exp(theta_logstd) + eps

        theta_mu_expected = torch.bmm(theta_mu, attn_sampled)
        theta_std_expected = torch.bmm(theta_std, attn_sampled)

        #theta sampled from N(theta_mu, theta_std)
        r_theta = rand_dist.sample((b, 1)).to(device)
        theta = (theta_std_expected*r_theta + theta_mu_expected).squeeze(2).squeeze(1)



        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta
        
        q_t = F.log_softmax(attn.view(b, -1), dim=1).view(b, attn.shape[2], attn.shape[3]) # B x R x H x W
        
        z_mu = z_mu.view(b, z_dim, attn.shape[2], attn.shape[3])
        z_std = z_std.view(b, z_dim, attn.shape[2], attn.shape[3])
        q_t_temp = q_t.unsqueeze(1).expand(b, z_dim, attn.shape[2], attn.shape[3])
        # to prevent kl_z causing a nan value, where q(t,r) becomes zero
        z_mu = torch.where(torch.exp(q_t_temp) == 0, torch.zeros_like(q_t_temp), z_mu)
        z_std = torch.where(torch.exp(q_t_temp) == 0, torch.ones_like(q_t_temp), z_std)
        q_z_given_t = Normal(z_mu, z_std) 
        
        theta_mu = theta_mu.view(b, attn.shape[2], attn.shape[3])
        theta_std = theta_std.view(b, attn.shape[2], attn.shape[3])
        # to prevent kl_theta causing a nan value, where q(t,r) becomes zero
        theta_mu = torch.where(torch.exp(q_t) == 0, torch.zeros_like(q_t), theta_mu)
        theta_std = torch.where(torch.exp(q_t) == 0, torch.ones_like(q_t), theta_std)
        q_theta_given_t = Normal(theta_mu, theta_std)
        
        

        # normal prior over t
        p_t_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([0.1]).to(device))
        p_t = p_t_dist.log_prob(x_translated_sample).sum(1).view(attn.shape[2], attn.shape[3]).unsqueeze(0)
        p_t = F.log_softmax(p_t.view(-1), dim=0).view(1, attn.shape[2], attn.shape[3])
 
        val1 = (torch.exp(q_t)*(q_t - p_t)).view(b, -1).sum(1)  # 
        
        prior_z = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        kl_z = kl_divergence(q_z_given_t, prior_z).sum(1) 
        
        prior_theta_given_t = Normal(torch.tensor([0.0]).to(device), torch.tensor([theta_prior]).to(device))
        kl_theta = kl_divergence(q_theta_given_t, prior_theta_given_t)

        val2 = (torch.exp(q_t) * (kl_theta + kl_z)).view(b, -1).sum(1)
        
        kl_div = val1 + val2
        kl_div = kl_div.mean()
        

    else:
        rand_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        
        attn, q_t_r, p_r, attn_sampled, offsets, theta_vals, z_vals = encoder_model(y)
        attn_sampled_over_locs = torch.sum(attn_sampled, dim=1).view(attn_sampled.shape[0], -1, 1)
        attn_sampled = attn_sampled.view(attn_sampled.shape[0], -1).unsqueeze(2)
        z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
        theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)

        z_dim = z_vals.size(1) // 2
        z_mu = z_vals[:,:z_dim, ]
        z_logstd = z_vals[:, z_dim:, ]
        z_std = torch.exp(z_logstd) 
        z_mu_expected = torch.bmm(z_mu, attn_sampled)
        z_std_expected = torch.bmm(z_std, attn_sampled)
        # draw samples from variational posterior to calculate
        r_z = rand_dist.sample((b, z_dim)).to(device)
        z = (z_std_expected*r_z + z_mu_expected).squeeze(2)
        
        attn_dim = attn.shape[3]
        if  attn_dim % 2:
            x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)
            y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)[::-1]
        else:
            x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)
            y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)[::-1]
        x_0,x_1 = np.meshgrid(x_grid, y_grid)
        x_translated_sample = np.stack([x_0.ravel(), x_1.ravel()], 1)
        x_translated_sample = torch.from_numpy(x_translated_sample).to(device)
        x_translated_batch = x_translated_sample.expand(b, x_translated_sample.size(0), x_translated_sample.size(1))
        x_translated_batch = x_translated_batch.transpose(1, 2)

        dx = torch.bmm(x_translated_batch.type(torch.float), attn_sampled_over_locs).squeeze(2).unsqueeze(1)
        x = x - dx # translate coordinates
        
        
        eps = 1e-6
        
        theta_mu = theta_vals[:, 0:1, ]
        theta_logstd = theta_vals[:, 1:2, ] 
        theta_std = torch.exp(theta_logstd) + eps
        theta_mu_expected = torch.bmm(theta_mu, attn_sampled)
        theta_std_expected = torch.bmm(theta_std, attn_sampled)
        r_theta = rand_dist.sample((b, 1)).to(device)
        theta = (theta_std_expected*r_theta + theta_mu_expected).squeeze(2).squeeze(1) 

        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta
        
        
        z_mu = z_mu.view(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3])
        z_std = z_std.view(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3])
        q_t_r_temp = q_t_r.unsqueeze(1).expand(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3])
        # to prevent kl_z causing a nan value, where q(t,r) becomes zero
        z_mu = torch.where(torch.exp(q_t_r_temp) == 0, torch.zeros_like(q_t_r_temp), z_mu)
        z_std = torch.where(torch.exp(q_t_r_temp) == 0, torch.ones_like(q_t_r_temp), z_std)
        q_z_given_t_r = Normal(z_mu, z_std) # B x z_dim x R x HW
        
        theta_mu = theta_mu.view(b, attn.shape[1], attn.shape[2], attn.shape[3])
        theta_std = theta_std.view(b, attn.shape[1], attn.shape[2], attn.shape[3])
        # to prevent kl_theta causing a nan value, where q(t,r) becomes zero
        theta_mu = torch.where(torch.exp(q_t_r) == 0, torch.zeros_like(q_t_r), theta_mu)
        theta_std = torch.where(torch.exp(q_t_r) == 0, torch.ones_like(q_t_r), theta_std)
        q_theta_given_t_r = Normal(theta_mu, theta_std)
            
        # normal prior over t
        p_t_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([0.1]).to(device))
        p_t = p_t_dist.log_prob(x_translated_sample).sum(1).view(attn.shape[2], attn.shape[3]).unsqueeze(0).unsqueeze(1)
        
        p_t_r = p_t + p_r.unsqueeze(0)
        p_t_r = F.log_softmax(p_t_r.view(-1), dim=0).view(1, attn.shape[1], attn.shape[2], attn.shape[3])
        
        val1 = (torch.exp(q_t_r)*(q_t_r - p_t_r)).view(b, -1).sum(1)  # 
        
        prior_z = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        kl_z = kl_divergence(q_z_given_t_r, prior_z).sum(1) 
        
        if groupconv >= 1:
            theta_prior_given_r = np.pi/groupconv
        else:
            theta_prior_given_r = theta_prior
        
        p_theta_given_t_r = Normal(offsets.unsqueeze(1).unsqueeze(2).to(device), torch.tensor([theta_prior_given_r]*attn.shape[1]).unsqueeze(1).unsqueeze(2).to(device))
        kl_theta = kl_divergence(q_theta_given_t_r, p_theta_given_t_r)
        
        val2 = torch.exp(q_t_r) * (kl_theta + kl_z)
        val2 = val2.view(b, -1).sum(1)
        
        kl_div = val1 + val2 
        kl_div = kl_div.mean()
        
        
        
    # reconstruct
    y_hat = generator_model(x.contiguous(), z).view(b, -1)

    y = y.view(b, -1)
    
    y_mu = y_hat
    y_var = None
    
    # if decoder generates one value for each pixel or outputs a (mean, std) for each pixel!
    if y_hat.size(1) > y.size(1):
        y_mu = y_hat[:,:y.size(1)]
        y_logvar = y_hat[:,y.size(1):]
        y_var = torch.exp(y_logvar)
    
    if ctf is not None: # apply the CTF filter
        pad = ctf.size(2)//2
        y_mu = y_mu.view(1, -1, n, n)
        y_mu = F.conv2d(y_mu, ctf, padding=pad, groups=ctf.size(0))
        y_mu = y_mu.view(-1, n*n)

        if y_var is not None:
            y_var = y_var.view(-1, 1, n, n)
            y_var = F.conv2d(y_var, ctf, padding=pad)
            y_var = y_var.view(-1, n*n)
    
    mask = None
    if mask_radius > 0:
        radius = mask_radius
        x_img = np.arange(-n//2, n//2, 1)
        y_img = np.arange(n//2, -n//2, -1)
        x_img_crd, y_img_crd = np.meshgrid(x_img, y_img)
        img_grid_sample = np.stack([x_img_crd.ravel(), y_img_crd.ravel()], 1)
        img_grid_sample = torch.from_numpy(img_grid_sample).to(device)
        
        img_grid_batch = img_grid_sample.expand(b, img_grid_sample.size(0), img_grid_sample.size(1)).cpu().numpy()
        
        center = dx.detach().cpu().numpy() / btw_pixels_space
        dist = np.sqrt((center[:, :, 0] - img_grid_batch[:, :, 0])**2 + (center[:, :, 1] - img_grid_batch[:, :, 1])**2)
        
        mask = torch.from_numpy(dist) < radius
        mask = mask.view(b, -1).to(device)
        

    if mask is not None:
        y = torch.where(mask, y, torch.zeros_like(y))
        y_mu = torch.where(mask, y_mu, torch.zeros_like(y_mu))

        if y_var is not None:
            y_var = y_var[mask]
            y_logvar = y_logvar[mask]    
    
    if y_var is not None:
        log_p_x_g_z = -0.5*torch.sum((y_mu - y)**2/y_var + y_logvar, 1).mean()
    else:
        log_p_x_g_z = -0.5*torch.sum((y_mu - y)**2, 1).mean()

    
    elbo = log_p_x_g_z - kl_div

    return elbo, log_p_x_g_z, kl_div




def train_epoch(iterator, x_coord, generator_model, encoder_model, optim, translation_inference, rotation_inference
                , epoch, num_epochs, N, device, params, theta_prior, groupconv
                , padding, batch_size, mask_radius):

    generator_model.train()
    encoder_model.train()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0
    ideal_batch_size = 100
    
    if batch_size < ideal_batch_size:
        counter = 0
        iterations = ideal_batch_size // batch_size
        elbo =0 
        log_p_x_g_z = 0
        kl_div = 0
        for mb in iterator:
            if len(mb) > 1:
                y,ctf = mb
            else:
                y = mb[0]
                ctf = None
            
            b = y.size(0)
            x = Variable(x_coord)
            y = Variable(y)
        
            l1, l2, l3 = eval_minibatch(x, y, ctf, generator_model, encoder_model, translation_inference, rotation_inference
                                        , epoch, device, theta_prior, groupconv, padding, mask_radius)
            (-l1*(1/iterations)).backward()
            log_p_x_g_z += l2.item()
            kl_div += l3.item()
            elbo += l1.item()
            
            counter += 1
            if counter == iterations:
                optim.step()
                optim.zero_grad()

                elbo = elbo / iterations
                gen_loss = -log_p_x_g_z / iterations
                kl_loss = kl_div / iterations

                c += (b*iterations)
                delta = (b*iterations)*(gen_loss - gen_loss_accum)
                gen_loss_accum += delta/c

                delta = (b*iterations)*(elbo - elbo_accum)
                elbo_accum += delta/c

                delta = (b*iterations)*(kl_loss - kl_loss_accum)
                kl_loss_accum += delta/c

                template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Error={:.5f}, KL={:.5f}'
                line = template.format(epoch+1, num_epochs, c/N, elbo_accum, gen_loss_accum
                                      , kl_loss_accum)
                print(line, end='\r', file=sys.stderr)
                
                elbo = 0
                log_p_x_g_z = 0
                kl_div = 0
                counter = 0
                
    else:
        for mb in iterator:
            if len(mb) > 1:
                y,ctf = mb
            else:
                y = mb[0]
                ctf = None
                
            b = y.size(0)
            x = Variable(x_coord)
            y = Variable(y)
            
            elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, ctf, generator_model, encoder_model, translation_inference
                                                       , rotation_inference, epoch, device, theta_prior
                                                       , groupconv, padding, mask_radius)

            loss = -elbo
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()
            optim.zero_grad()

            elbo = elbo.item()
            gen_loss = -log_p_x_g_z.item()
            kl_loss = kl_div.item()

            c += b
            delta = b*(gen_loss - gen_loss_accum)
            gen_loss_accum += delta/c

            delta = b*(elbo - elbo_accum)
            elbo_accum += delta/c

            delta = b*(kl_loss - kl_loss_accum)
            kl_loss_accum += delta/c

            template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Error={:.5f}, KL={:.5f}'
            line = template.format(epoch+1, num_epochs, c/N, elbo_accum, gen_loss_accum
                                  , kl_loss_accum)
            print(line, end='\r', file=sys.stderr)

    print(' '*150, end='\r', file=sys.stderr)
    return elbo_accum, gen_loss_accum, kl_loss_accum




def eval_model(iterator, x_coord, generator_model, encoder_model, translation_inference , rotation_inference, epoch
               , device, theta_prior, groupconv, padding, mask_radius):
    generator_model.eval()
    encoder_model.eval()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0
    
    with torch.no_grad():
        for mb in iterator:
            if len(mb) > 1:
                y,ctf = mb
            else:
                y = mb[0]
                ctf = None
                
            b = y.size(0)
            x = Variable(x_coord)
            y = Variable(y)

            elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, ctf, generator_model, encoder_model, translation_inference
                                                       , rotation_inference, epoch, device, theta_prior
                                                       , groupconv, padding, mask_radius)

            elbo = elbo.item()
            gen_loss = -log_p_x_g_z.item()
            kl_loss = kl_div.item()

            c += b
            delta = b*(gen_loss - gen_loss_accum)
            gen_loss_accum += delta/c

            delta = b*(elbo - elbo_accum)
            elbo_accum += delta/c

            delta = b*(kl_loss - kl_loss_accum)
            kl_loss_accum += delta/c

    return elbo_accum, gen_loss_accum, kl_loss_accum




def load_images(path):
    if path.endswith('mrc') or path.endswith('mrcs'):
        with open(path, 'rb') as f:
            content = f.read()
        images,_,_ = mrc.parse(content)
    elif path.endswith('npy'):
        images = np.load(path)
    return images


class Dataset:
    def __init__(self, y, ctf=None):
        self.y = y
        self.ctf = ctf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.ctf is None:
            return self.y[i], None
        return self.y[i], self.ctf[i]


def main():
    import argparse

    parser = argparse.ArgumentParser('Training on particle datasets')

    parser.add_argument('--train-path', help='path to training data; or path to the whole data')
    parser.add_argument('--test-path', help='path to testing data')
    parser.add_argument('--ctf-train', help='path to CTF parameters for training images;or path to CTF parameters of whole set')
    parser.add_argument('--ctf-test', help='path to CTF parameters for testing images')
    parser.add_argument('--scale', default=1, type=float, help='used to scale the ang/pix if images were binned (default: 1)')
    parser.add_argument('--in-channels', type=int, default=1, help='number of channels in the images')
    
    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    
    parser.add_argument('--t-inf', default='unimodal', choices=['unimodal', 'attention'], help='unimodal | attention')
    parser.add_argument('--r-inf', default='unimodal', choices=['unimodal', 'attention', 'attention+offsets'], help='unimodal | attention | attention+offsets')
    parser.add_argument('--groupconv', type=int, default=0, choices=[0, 4, 8, 16], help='0 | 4 | 8 | 16')
    parser.add_argument('--encoder-num-layers', type=int, default=2, help='number of hidden layers in original spatial-VAE inference model')
    parser.add_argument('--encoder-kernel-number', type=int, default=128, help='number of kernels in each layer of the encoder (default: 128)')
    parser.add_argument('--encoder-kernel-size', type=int, default=64, help='size of kernels in the first layer of the encoder when using conv/groupconv layers (default: 64)')
    parser.add_argument('--encoder-padding', type=int, default=16, help='amount of the padding for the encoder (default: 16)')
    
    parser.add_argument('--fourier-expansion', action='store_true', help='using random fourier feature expansion in generator!')
    parser.add_argument('--fourier-sigma', type=float, default=0.01, help='sigma value for random fourier feature expansion (default:0.01)')
    
    parser.add_argument('--generator-hidden-dim', type=int, default=512, help='dimension of hidden layers (default: 500)')
    parser.add_argument('--generator-num-layers', type=int, default=2, help='number of hidden layers (default: 2)')
    parser.add_argument('--generator-resid-layers', action="store_true", help='using skip connections in generator')
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')
    
    
    parser.add_argument('-l', '--learning-rate', type=float, default=2e-4, help='learning rate (default: 0.001)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')
    parser.add_argument('--train-portion', default=0.9, type=float, help='portion of dataset used for training (default: 0.8)')

    parser.add_argument('--log-root', default='./training_logs', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=20, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=500, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')

    parser.add_argument('--fit-noise', action='store_true', help='also learn the standard deviation of the noise in the generative model')
    parser.add_argument('--softplus', action='store_true', help='apply softplus activation to mean pixel output by generator. clamping the mean to be non-negative can reduce learning background noise')
    
    parser.add_argument('--normalize', action='store_true', help='normalize the images before training')
    parser.add_argument('--mask-radius', default=0, type=int, help='radius of the circular mask for the reconstructed images')
    parser.add_argument('--crop', default=0, type=int, help='size of the cropped images')

    args = parser.parse_args()
    num_epochs = args.num_epochs

    digits = int(np.log10(num_epochs)) + 1
    scale = args.scale
    mask_radius = args.mask_radius
    crop = args.crop
    ctf_train = None
    ctf_test = None
    

    if args.train_path and args.test_path:
        images_train = load_images(args.train_path)
        images_test = load_images(args.test_path)
        n,m = images_train.shape[1:]
        
        if args.ctf_train and args.ctf_test:
            print('# loading CTF filters:', args.ctf_train, file=sys.stderr)
            if n % 2 == 0: ctf_n = n - 1
            if m % 2 == 0: ctf_m = m - 1
            ctf_params = C.parse_ctf(args.ctf_train)
            ctf_train = C.ctf_filter(ctf_params, ctf_n, ctf_m, scale=scale)
            ctf_train = torch.from_numpy(ctf_train).float().unsqueeze(1)
            
            ctf_params = C.parse_ctf(args.ctf_test)
            ctf_test = C.ctf_filter(ctf_params, ctf_n, ctf_m, scale=scale)
            ctf_test = torch.from_numpy(ctf_test).float().unsqueeze(1)
        
    elif args.train_path:
        images = load_images(args.train_path)
        
        train_size = int(images.shape[0] * args.train_portion)
        images_train = images[:train_size,:,:]
        images_test = images[train_size:, :, :]
        n,m = images_train.shape[1:]
        
        if args.ctf_train:
            print('# loading CTF filters:', args.ctf_train, file=sys.stderr)
            if n % 2 == 0: 
                ctf_n = n - 1
            if m % 2 == 0: 
                ctf_m = m - 1
            
            ctf_params = C.parse_ctf(args.ctf_train)
            ctf_filters = C.ctf_filter(ctf_params, ctf_n, ctf_m, scale=scale)
            
            ctf_train = ctf_filters[:train_size, : ]
            ctf_train = torch.from_numpy(ctf_train).float().unsqueeze(1)
            print('*')
            print(ctf_train.shape)
            ctf_test = ctf_filters[train_size:, : ]
            ctf_test = torch.from_numpy(ctf_test).float().unsqueeze(1)
            print('*')
            print(ctf_test.shape)
        
    else:
        print('please provide the train_path and/or test_path', file=sys.stderr)
        return
    
    if crop > 0:
        images_train = image_utils.crop(images_train, crop)
        images_test = image_utils.crop(images_test, crop)
        print('# cropped to:', crop, file=sys.stderr)
    
    n,m = images_train.shape[1:]
    
    # normalize the images using edges to estimate background
    if args.normalize:
        print('# normalizing particles', file=sys.stderr)
        mu = images_train.reshape(-1, n*m).mean(1)
        std = images_train.reshape(-1, n*m).std(1)
        images_train = (images_train - mu[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]

        mu = images_test.reshape(-1, n*m).mean(1)
        std = images_test.reshape(-1, n*m).std(1)
        images_test = (images_test - mu[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
    
    
    ## x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    images_train = torch.from_numpy(images_train).float()
    images_test = torch.from_numpy(images_test).float()

    in_channels = args.in_channels
    y_train = images_train.view(-1, in_channels, n, m)
    y_test = images_test.view(-1, in_channels, n, m)
    

    ## set the device
    device = args.device
    use_cuda = (device != -1) and torch.cuda.is_available()
    if device >= 0:
        torch.cuda.set_device(device)
        print('# using CUDA device:', device, file=sys.stderr)

    
    if use_cuda:
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        if ctf_train is not None:
            ctf_train = ctf_train.cuda()
        if ctf_test is not None:
            ctf_test = ctf_test.cuda()

    if use_cuda:
        x_coord = x_coord.cuda()


    data_train = torch.utils.data.TensorDataset(y_train)
    if ctf_train is not None:
        data_train = torch.utils.data.TensorDataset(y_train, ctf_train)

    data_test = torch.utils.data.TensorDataset(y_test)
    if ctf_test is not None:
        data_test = torch.utils.data.TensorDataset(y_test, ctf_test)

    z_dim = args.z_dim
    print('# training with z-dim:', z_dim, file=sys.stderr)

    generator_num_layers = args.generator_num_layers
    generator_hidden_dim = args.generator_hidden_dim
    generator_resid = args.generator_resid_layers
    
    fourier_expansion = args.fourier_expansion
    fourier_sigma = args.fourier_sigma
    if fourier_expansion:
        print('# Using random Fourier feature expansion', file=sys.stderr)

    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'leakyrelu':
        activation = nn.LeakyReLU
    
    softplus = args.softplus
    fit_noise = args.fit_noise
    n_out = 1
    if fit_noise:
        n_out = 2
    # defining generator_model
    generator_model = models.SpatialGenerator(z_dim, generator_hidden_dim, n_out=n_out, num_layers=generator_num_layers
                                              , activation=activation, softplus=softplus, resid=generator_resid
                                              , fourier_expansion=fourier_expansion, sigma=fourier_sigma)

    # defining encoder_model model
    translation_inference = args.t_inf
    rotation_inference = args.r_inf
    encoder_num_layers = args.encoder_num_layers
    encoder_kernel_number = args.encoder_kernel_number
    encoder_kernel_size = args.encoder_kernel_size
    encoder_padding = args.encoder_padding
    group_conv = args.groupconv

    print('# translation inference is {}'.format(translation_inference), file=sys.stderr)
    print('# rotation inference is {}'.format(rotation_inference), file=sys.stderr)
    
    theta_prior = np.pi
    normal_prior_over_r = False
    print('# using priors: theta={}'.format(theta_prior), file=sys.stderr)
    
    
    if translation_inference=='unimodal' and rotation_inference=='unimodal': #original spatial-vae from Bepler et. al 2019
        inf_dim = z_dim + 3 # 1 additional dim for rotation and 2 for translation 
        encoder_model = models.InferenceNetwork_UnimodalTranslation_UnimodalRotation(m*n, inf_dim, encoder_kernel_number, num_layers=encoder_num_layers, activation=activation)

    elif translation_inference=='attention' and rotation_inference=='unimodal':
        encoder_model = models.InferenceNetwork_AttentionTranslation_UnimodalRotation(m, in_channels, z_dim, kernels_num=encoder_kernel_number, activation=activation, groupconv=group_conv)

    elif translation_inference=='attention' and (rotation_inference=='attention' or rotation_inference=='attention+offsets'):
        rot_refinement = (rotation_inference=='attention+offsets')
        encoder_model = models.InferenceNetwork_AttentionTranslation_AttentionRotation(m, in_channels, z_dim, kernels_num=encoder_kernel_number, kernels_size=encoder_kernel_size, padding=encoder_padding, activation=activation, groupconv=group_conv, rot_refinement=rot_refinement, theta_prior=theta_prior, normal_prior_over_r=normal_prior_over_r)
    

    generator_model.to(device)
    encoder_model.to(device)
    print(encoder_model)
    print(generator_model)
    

    N = len(y_train)

    params = list(generator_model.parameters()) + list(encoder_model.parameters())
    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)
    
    scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=10, threshold=1e-4, threshold_mode='abs', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)
    

    minibatch_size = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size, shuffle=True)
    test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)

    output = sys.stdout
    print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']), file=output)


    #creating the log folder
    log_root = args.log_root
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    experiment_description = '_'.join([str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
                                       , args.train_path.replace('/', '-'), 'zDim', str(z_dim)
                                       , 'translation',translation_inference, 'rotation', rotation_inference])
    if group_conv > 0:
        experiment_description = experiment_description + '_groupconv' + str(group_conv)
    if args.ctf_train:
        experiment_description = experiment_description + '_ctf'
    if fourier_expansion:
        experiment_description = experiment_description + '_Fr_sigma' + str(fourier_sigma)
        
    path_prefix = os.path.join(log_root, experiment_description,'')

    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)   

    save_interval = args.save_interval
    train_log = ['' for _ in range(3*num_epochs)]
    
    print('# learning-rate is {}'.format(lr), file=sys.stderr)
    
    
    early_stopping = EarlyStopping(patience=20, delta=1e-4, save_path=path_prefix, digits=digits)

    
    for epoch in range(num_epochs):
        
        elbo_accum, gen_loss_accum, kl_loss_accum = train_epoch(train_iterator, x_coord, generator_model, encoder_model, optim
                                                              , translation_inference=translation_inference
                                                              , rotation_inference=rotation_inference, epoch=epoch
                                                              , num_epochs=num_epochs, N=N, device=device, params=params
                                                              , theta_prior=theta_prior, groupconv=group_conv
                                                              , padding = encoder_padding, batch_size=minibatch_size
                                                              , mask_radius=mask_radius)

        line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        train_log[3*epoch] = line
        print(line, file=output)
        

        # evaluate on the test set
        elbo_accum, gen_loss_accum, kl_loss_accum = eval_model(test_iterator, x_coord, generator_model, encoder_model
                                                             , translation_inference=translation_inference
                                                             , rotation_inference=rotation_inference, epoch=epoch
                                                             , device=device, theta_prior=theta_prior, groupconv=group_conv
                                                             , padding = encoder_padding, mask_radius=mask_radius)
        
        
        
        line = '\t'.join([str(epoch+1), 'test', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        train_log[(3*epoch)+1] = line
        print(line, file=output)
        
        
        
        # checking for early stopping
        line = early_stopping(elbo_accum, encoder_model, generator_model, epoch+1)
        train_log[(3*epoch)+2] = line
        print(line, file=output)
        print('\n', file=output)
        output.flush()
        if early_stopping.early_stop:
            print("*** Early stopping ***")
            break
        
        generator_model.to(device)
        encoder_model.to(device)
        
        scheduler.step(elbo_accum)
        
        ## save the models
        if (epoch+1)%save_interval == 0:
            epoch_str = str(epoch+1).zfill(digits)

            path = path_prefix + 'generator_epoch{}.sav'.format(epoch_str)
            generator_model.eval().cpu()
            torch.save(generator_model, path)

            path = path_prefix + 'inference_epoch{}.sav'.format(epoch_str)
            encoder_model.eval().cpu()
            torch.save(encoder_model, path)

            generator_model.to(device)
            encoder_model.to(device)
        
    with open(path_prefix + 'train_log.txt', 'w') as f:
        f.write(experiment_description + '\n')
        f.write('\n\nargs:')
        f.write(str(args))
        f.write('\n\n')
        
        f.write('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']) + '\n')
        for i in range(3*(epoch+1)):
            if i > 0 and (i%3 == 0):
                f.write('\n')
            f.write(train_log[i] + '\n')
            
            
        f.write('Encoder model: \n {}'.format(encoder_model))
        f.write('\nGenerator model: \n {}'.format(generator_model))
        

if __name__ == '__main__':
    main()
