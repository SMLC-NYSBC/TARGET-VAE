from __future__ import print_function, division

import numpy as np
import sys

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
from src.utils import EarlyStopping

import datetime
import os
import shutil




def eval_minibatch(x, y, generator_model, encoder_model, t_inf, r_inf, epoch, device
                  , theta_prior, groupconv, image_dim):

    b = y.size(0)
    btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()
    x = x.expand(b, x.size(0), x.size(1)).to(device)

    y = y.to(device)

    if t_inf == 'unimodal' and r_inf == 'unimodal':
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


    elif t_inf == 'attention' and r_inf == 'unimodal':
        rand_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        
        attn, attn_sampled, theta_vals, z_vals = encoder_model(y, device)
        
        #attn_sampled returned here is over the locations since r_inf is unimodal
        attn_sampled = attn_sampled.view(attn_sampled.shape[0], -1).unsqueeze(2)
        z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
        theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)


        eps = 1e-6
        
        z_dim = z_vals.size(1) // 2
        z_mu = z_vals[:,:z_dim, ]
        z_logstd = z_vals[:, z_dim:, ]
        z_std = torch.exp(z_logstd) + eps

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
        
        attn, q_t_r, p_r, attn_sampled, offsets, theta_vals, z_vals = encoder_model(y, device)

        attn_sampled_over_locs = torch.sum(attn_sampled, dim=1).view(attn_sampled.shape[0], -1, 1)
        attn_sampled = attn_sampled.view(attn_sampled.shape[0], -1).unsqueeze(2)
        z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
        theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)

        eps = 1e-6
        
        z_dim = z_vals.size(1) // 2
        z_mu = z_vals[:,:z_dim, ]
        z_logstd = z_vals[:, z_dim:, ]
        z_std = torch.exp(z_logstd) + eps
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
        
        p_theta_given_t_r = Normal(offsets.unsqueeze(1).unsqueeze(2).to(device), 
                                   torch.tensor([theta_prior_given_r]*attn.shape[1]).unsqueeze(1).unsqueeze(2).to(device))
        kl_theta = kl_divergence(q_theta_given_t_r, p_theta_given_t_r)
        
        val2 = torch.exp(q_t_r) * (kl_theta + kl_z)
        val2 = val2.view(b, -1).sum(1)
        
        kl_div = val1 + val2 
        kl_div = kl_div.mean()
        
        
        
    # reconstruct
    y_hat = generator_model(x.contiguous(), z)
    y_hat = y_hat.view(b, -1)
    y = y.view(b, -1)
    size = y.size(1)
    log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y)*size
    elbo = log_p_x_g_z - kl_div
    
    return elbo, log_p_x_g_z, kl_div





def train_epoch(iterator, x_coord, generator_model, encoder_model, optim, t_inf, r_inf
                , epoch, num_epochs, N, device, params, theta_prior, groupconv, image_dim):

    generator_model.train()
    encoder_model.train()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0
    
    for y, in iterator:
        b = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, generator_model, encoder_model, t_inf, 
                            r_inf, epoch, device, theta_prior, groupconv, image_dim)

        
        loss = -elbo
        loss.backward()

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





def eval_model(iterator, x_coord, generator_model, encoder_model, t_inf , r_inf, epoch
               , device, theta_prior, groupconv, image_dim):

    generator_model.eval()
    encoder_model.eval()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0
    
    with torch.no_grad():
        for y, in iterator:
            b = y.size(0)
            x = Variable(x_coord)
            y = Variable(y)

            elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, generator_model, encoder_model, t_inf
                                                       , r_inf, epoch, device, theta_prior
                                                       , groupconv,image_dim)

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










def main():
    import argparse

    parser = argparse.ArgumentParser('Train TARGET_VAE on MNIST/MNIST-N/MNIST-U datasets')
    
    parser.add_argument('--dataset', choices=['mnist', 'mnist-U', 'mnist-N'], default='mnist-U', help='MNIST datset to train/validate(default: mnist-U)')
    
    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--t-inf', default='attention', choices=['unimodal', 'attention'], help='unimodal | attention (default: attention)')
    
    parser.add_argument('--r-inf', default='attention+offsets', choices=['unimodal', 'attention', 'attention+offsets'], help='unimodal | attention | attention+offsets (default: attention+offsets)')
    
    parser.add_argument('--groupconv', type=int, default=8, choices=[0, 4, 8, 16], help='0 | 4 | 8 | 16 (default:8)')
    parser.add_argument('--encoder-num-layers', type=int, default=2, help='number of hidden layers in the inference model when the translation and rotation inference are unimodal (default:2)')
    parser.add_argument('--encoder-kernel-number', type=int, default=128, help='number of kernels in each layer of the encoder (default: 128)')
    parser.add_argument('--encoder-kernel-size', type=int, default=28, help='size of kernels in the first layer of the encoder (default: 28)')
    parser.add_argument('--encoder-padding', type=int, default=8, help='amount of the padding for the encoder (default: 8)')
    
    parser.add_argument('--in-channels', type=int, default=1, help='number of channels in the images (default:1)')
    parser.add_argument('--image-dim', type=int, default=50, help='input image of the shape image_dim x image_dim (default:50)')
    parser.add_argument('--fourier-expansion', action='store_true', help='using random fourier feature expansion in generator')

    parser.add_argument('--generator-hidden-dim', type=int, default=512, help='dimension of hidden layers (default: 512)')
    parser.add_argument('--generator-num-layers', type=int, default=2, help='number of hidden layers (default: 2)')
    parser.add_argument('--generator-resid-layers', action="store_true", help='using skip connections in generator')
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')

    parser.add_argument('-l', '--learning-rate', type=float, default=2e-4, help='learning rate (default: 2e-4)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--log-root', default='./training_logs', help='path prefix to save models (default:./training_logs)')
    parser.add_argument('--save-interval', default=20, type=int, help='save frequency in epochs (default: 20)')
    parser.add_argument('--num-epochs', type=int, default=500, help='number of training epochs (default: 500)')

    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use (default:0)')

    args = parser.parse_args()
    num_epochs = args.num_epochs

    digits = int(np.log10(num_epochs)) + 1
    
    ## load the images
    if args.dataset == 'mnist':
        print('# training on MNIST', file=sys.stderr)
        mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST('data/', train=False, download=True)

        array = np.zeros((len(mnist_train), args.image_dim, args.image_dim), dtype=np.uint8)
        for i in range(len(mnist_train)):
            array[i] = np.array(mnist_train[i][0], copy=False)
        mnist_train = array

        array = np.zeros((len(mnist_test), args.image_dim, args.image_dim), dtype=np.uint8)
        for i in range(len(mnist_test)):
            array[i] = np.array(mnist_test[i][0], copy=False)
        mnist_test = array

    elif args.dataset == 'mnist-U':
        print('# training on rotated and translated MNIST with uniform rotations', file=sys.stderr)
        mnist_train = np.load('data/mnist_U/images_train.npy')
        mnist_test = np.load('data/mnist_U/images_test.npy')
    
    elif args.dataset == 'mnist-N':
        print('# training on rotated and translated MNIST with normal rotations', file=sys.stderr)
        mnist_train = np.load('data/mnist_N/images_train.npy')
        mnist_test = np.load('data/mnist_N/images_test.npy')
        
    else:
        print('# The dataset does not exist!', file=sys.stderr)
        return
    
    mnist_train = torch.from_numpy(mnist_train).float()/255
    mnist_test = torch.from_numpy(mnist_test).float()/255
    
    image_dim = args.image_dim

    ## x coordinate array
    xgrid = np.linspace(-1, 1, image_dim)
    ygrid = np.linspace(1, -1, image_dim)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    in_channels = args.in_channels
    y_train = mnist_train.view(-1, in_channels, image_dim, image_dim)
    y_test = mnist_test.view(-1, in_channels, image_dim, image_dim)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(d)
        print('# using CUDA device:', d, file=sys.stderr)
        device = torch.device("cuda:" + str(d) if use_cuda else "cpu")
    else:
        device = torch.device("cpu")

    y_train = y_train.to(device)
    y_test = y_test.to(device)
    x_coord = x_coord.to(device)

    data_train = torch.utils.data.TensorDataset(y_train)
    data_test = torch.utils.data.TensorDataset(y_test)

    
    z_dim = args.z_dim
    print('# training with z-dim:', z_dim, file=sys.stderr)

    generator_num_layers = args.generator_num_layers
    generator_hidden_dim = args.generator_hidden_dim
    generator_resid = args.generator_resid_layers
    
    fourier_expansion = args.fourier_expansion
    fourier_sigma = max(2.0 / (image_dim - 1), 2.0 / (image_dim - 1)) # setting sigma value of the fourier expansion to the pixel size in the image
    if fourier_expansion:
        print('# Using random Fourier feature expansion', file=sys.stderr)
        print('# The sigma value for the Fourier feature expansion is {}'.format(fourier_sigma), file=sys.stderr)

    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'leakyrelu':
        activation = nn.LeakyReLU

    # defining generator_model
    generator_model = models.SpatialGenerator(z_dim, generator_hidden_dim, num_layers=generator_num_layers, activation=activation
                                              , resid=generator_resid, fourier_expansion=fourier_expansion, sigma=fourier_sigma)

    # defining encoder_model model
    t_inf = args.t_inf
    r_inf = args.r_inf
    encoder_num_layers = args.encoder_num_layers
    encoder_kernel_number = args.encoder_kernel_number
    encoder_kernel_size = args.encoder_kernel_size
    encoder_padding = args.encoder_padding
    group_conv = args.groupconv

    print('# translation inference is {}'.format(t_inf), file=sys.stderr)
    print('# rotation inference is {}'.format(r_inf), file=sys.stderr)
    
    
    if args.dataset == 'mnist-N':
        theta_prior = np.pi/4 #std for prior on theta
        normal_prior_over_r = True
    else:
        theta_prior = np.pi 
        normal_prior_over_r = False
    
    if normal_prior_over_r:
        print('# Gaussian prior over theta with mean=0 and std={}'.format(theta_prior), file=sys.stderr)
    else:
        print('# Uniform prior over theta', file=sys.stderr)
        
    
    if t_inf=='unimodal' and r_inf=='unimodal': 
        inf_dim = z_dim + 3 # 1 additional dim for rotation and 2 for translation 
        encoder_model = models.InferenceNetwork_UnimodalTranslation_UnimodalRotation(image_dim*image_dim, inf_dim, encoder_kernel_number
                                                                                     , num_layers=encoder_num_layers, activation=activation)

    elif t_inf=='attention' and r_inf=='unimodal':
        encoder_model = models.InferenceNetwork_AttentionTranslation_UnimodalRotation(image_dim, in_channels, z_dim, kernels_num=encoder_kernel_number
                                                                                      , activation=activation, groupconv=group_conv)

    elif t_inf=='attention' and (r_inf=='attention' or r_inf=='attention+offsets'):
        rot_refinement = (r_inf=='attention+offsets')
        encoder_model = models.InferenceNetwork_AttentionTranslation_AttentionRotation(image_dim, in_channels, z_dim, kernels_num=encoder_kernel_number
                                                                                       , kernels_size=encoder_kernel_size, padding=encoder_padding
                                                                                       , activation=activation, groupconv=group_conv
                                                                                       , rot_refinement=rot_refinement, theta_prior=theta_prior
                                                                                       , normal_prior_over_r=normal_prior_over_r)


    generator_model.to(device)
    encoder_model.to(device)
    print(encoder_model)
    print(generator_model)
    
    
    N = len(mnist_train)

    params = list(generator_model.parameters()) + list(encoder_model.parameters())
    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)
    
    scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=9, threshold=1e-4, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08
                                  , verbose=True)
    
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
                                       , args.dataset, 'zDim', str(z_dim),'translation', t_inf, 'rotation'
                                       , r_inf])
    if group_conv > 0:
        experiment_description = experiment_description + '_groupconv' + str(group_conv)
        
    path_prefix = os.path.join(log_root, experiment_description,'')

    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)   

    save_interval = args.save_interval
    train_log = ['' for _ in range(3*num_epochs)]
    
    print('# learning-rate is {}'.format(lr))
    
    
    early_stopping = EarlyStopping(patience=20, delta=1e-4, save_path=path_prefix, digits=digits)
    
    with open(path_prefix + 'train_log.txt', 'w', 1) as log_file:
        print(experiment_description + '\n', file=log_file)
        print('\n\nargs:', file=log_file)
        print(str(args), file=log_file)
        print('\nEncoder model: \n {}'.format(encoder_model), file=log_file)
        print('\nGenerator model: \n {}'.format(generator_model), file=log_file)
        
        print('\n\n', file=log_file)
        print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']) + '\n', file=log_file)
        
        for epoch in range(num_epochs):

            elbo_accum, gen_loss_accum, kl_loss_accum = train_epoch(train_iterator, x_coord, generator_model, encoder_model, optim
                                                                  , t_inf=t_inf
                                                                  , r_inf=r_inf, epoch=epoch
                                                                  , num_epochs=num_epochs, N=N, device=device, params=params
                                                                  , theta_prior=theta_prior, groupconv=group_conv
                                                                  , image_dim=image_dim)

            line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
            train_log[3*epoch] = line
            print(line, file=output)
            print(line, file=log_file)

            # evaluate on the test set
            elbo_accum, gen_loss_accum, kl_loss_accum = eval_model(test_iterator, x_coord, generator_model, encoder_model
                                                                 , t_inf=t_inf
                                                                 , r_inf=r_inf, epoch=epoch
                                                                 , device=device, theta_prior=theta_prior, groupconv=group_conv
                                                                 , image_dim=image_dim)


            line = '\t'.join([str(epoch+1), 'test', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
            train_log[(3*epoch)+1] = line
            print(line, file=output)
            print(line, file=log_file)


            # checking for early stopping
            line = early_stopping(elbo_accum, encoder_model, generator_model, epoch+1)
            train_log[(3*epoch)+2] = line
            print(line, file=output)
            print('\n', file=output)
            print(line, file=log_file)
            print('\n', file=log_file)
            
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
        



            
            
        


if __name__ == '__main__':
    main()





