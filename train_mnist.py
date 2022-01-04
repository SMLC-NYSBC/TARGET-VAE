from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

import src.models as models

import datetime
import os
import shutil




def eval_minibatch(x, y, generator_model, encoder_model, translation_inference, rotation_inference, epoch, device
                  , theta_prior):

    b = y.size(0)
    btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()
    x = x.expand(b, x.size(0), x.size(1)).to(device)

    y = y.to(device)

    if translation_inference == 'unimodal' and rotation_inference == 'unimodal':
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

        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta

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
        x = x + dx # translate coordinates

        # reconstruct
        y_hat = generator_model(x.contiguous(), z)
        y_hat = y_hat.view(b, -1)

        # unit normal prior over z and translation
        z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
        kl_div = kl_div + torch.sum(z_kl, 1)
        kl_div = kl_div.mean()


    elif translation_inference == 'attention' and rotation_inference == 'unimodal':
        rand_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        prior_rot = Normal(torch.tensor([0.0]).to(device), torch.tensor([theta_prior]).to(device))
        prior_z = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        probs, theta_vals, z_vals = encoder_model(y)

        probs = probs.view(probs.shape[0], -1).unsqueeze(2)
        z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
        theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)


        z_dim = z_vals.size(1) // 2
        z_mu = z_vals[:,:z_dim, ]
        z_logstd = z_vals[:, z_dim:, ]
        z_std = torch.exp(z_logstd)

        z_mu_expected = torch.bmm(z_mu, probs)
        z_std_expected = torch.bmm(z_std, probs)

        # draw samples from variational posterior to calculate
        r_z = rand_dist.sample((b, z_dim)).to(device)
        z = (z_std_expected*r_z + z_mu_expected).squeeze(2)

        #btw_pixels_space = 0.0741
        x_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)
        y_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)[::-1]
        x_0,x_1 = np.meshgrid(x_grid, y_grid)
        x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
        x_coord_translate = torch.from_numpy(x_coord_translate).float().to(device)
        x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
        x_coord_translate = x_coord_translate.transpose(1, 2)

        dx_expected = torch.bmm(x_coord_translate, probs).squeeze(2).unsqueeze(1)
        x = x - dx_expected # translate coordinates


        theta_mu = theta_vals[:, 0:1, ]
        theta_logstd = theta_vals[:, 1:2, ]
        theta_std = torch.exp(theta_logstd)

        theta_mu_expected = torch.bmm(theta_mu, probs)
        theta_std_expected = torch.bmm(theta_std, probs)

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


        q_z_given_t = Normal(z_mu.view(b, z_dim, attn.shape[1], attn.shape[2]), z_std.view(b, z_dim, attn.shape[1], attn.shape[2])) 
        q_theta_given_t = Normal(theta_mu.view(b, attn.shape[1], attn.shape[2]), theta_std.view(b, attn.shape[1], attn.shape[2]))
        q_t = F.log_softmax(attn.view(b, -1), dim=1).view(b, attn.shape[1], attn.shape[2]) # B x R x H x W


        # uniform prior over t
        #p_t = torch.zeros_like(q_t_r).to(device) - np.log(attn.shape[2]*attn.shape[3])

        # normal prior over t
        p_t = torch.zeros(1, 1, attn.shape[2], attn.shape[3]).to(device)
        p_t[:, :, :, :] = p_t_dist.log_prob(torch.tensor([x_grid]).to(device)).transpose(0, 1) + p_t_dist.log_prob(torch.tensor([y_grid]).to(device))
        p_t = p_t.expand(b, attn.shape[1], p_t.shape[2], p_t.shape[3]) 

        val1 = (torch.exp(q_t)*(q_t - p_t)).view(b, -1).sum(1)  # 


        kl_z = kl_divergence(q_z_given_t, prior_z).sum(1) 
        kl_theta = kl_divergence(q_theta_given_t, prior_rot)

        val2 = (torch.exp(q_t) * (kl_theta + kl_z)).view(b, -1).sum(1)

        kl_div = val1 + val2
        kl_div = kl_div.mean()

    else:
        rand_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        prior_rot = Normal(torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2]).unsqueeze(1).unsqueeze(2).to(device), torch.tensor([theta_prior]*4).unsqueeze(1).unsqueeze(2).to(device))
        prior_z = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        p_t_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([0.1]).to(device))

        attn, probs, theta_vals, z_vals = encoder_model(y, epoch)

        probs_over_locs = torch.sum(probs, dim=1).view(probs.shape[0], -1, 1)
        probs = probs.view(probs.shape[0], -1).unsqueeze(2)
        z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
        theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)

        z_dim = z_vals.size(1) // 2
        z_mu = z_vals[:,:z_dim, ]
        z_logstd = z_vals[:, z_dim:, ]
        z_std = torch.exp(z_logstd) 
        z_mu_expected = torch.bmm(z_mu, probs)
        z_std_expected = torch.bmm(z_std, probs)
        # draw samples from variational posterior to calculate
        r_z = rand_dist.sample((b, z_dim)).to(device)
        z = (z_std_expected*r_z + z_mu_expected).squeeze(2)

        #btw_pixels_space = 0.0741
        x_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)
        y_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)[::-1]
        x_0,x_1 = np.meshgrid(x_grid, y_grid)
        x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
        x_coord_translate = torch.from_numpy(x_coord_translate).to(device)
        x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
        x_coord_translate = x_coord_translate.transpose(1, 2)
        dx_expected = torch.bmm(x_coord_translate.type(torch.float), probs_over_locs).squeeze(2).unsqueeze(1)
        x = x - dx_expected # translate coordinates

        theta_mu = theta_vals[:, 0:1, ]
        theta_logstd = theta_vals[:, 1:2, ]
        theta_std = torch.exp(theta_logstd) 
        theta_mu_expected = torch.bmm(theta_mu, probs)
        theta_std_expected = torch.bmm(theta_std, probs)
        r_theta = rand_dist.sample((b, 1)).to(device)
        theta = (theta_std_expected*r_theta + theta_mu_expected).squeeze(2).squeeze(1) 

        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta


        q_z_given_t_r = Normal(z_mu.view(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3]), z_std.view(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3])) # B x z_dim x R x HW
        q_theta_given_t_r = Normal(theta_mu.view(b, attn.shape[1], attn.shape[2], attn.shape[3]), theta_std.view(b, attn.shape[1], attn.shape[2], attn.shape[3]))
        q_t_r = F.log_softmax(attn.view(b, -1), dim=1).view(b, attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W

        # uniform prior over r
        p_r_given_t = torch.zeros_like(q_t_r).to(device) - np.log(attn.shape[1])

        # uniform prior over t
        #p_t = torch.zeros_like(q_t_r).to(device) - np.log(attn.shape[2]*attn.shape[3])

        # normal prior over t
        p_t = torch.zeros(1, 1, attn.shape[2], attn.shape[3]).to(device)
        p_t[:, :, :, :] = p_t_dist.log_prob(torch.tensor([x_grid]).to(device)).transpose(0, 1) + p_t_dist.log_prob(torch.tensor([y_grid]).to(device))
        p_t = p_t.expand(b, attn.shape[1], p_t.shape[2], p_t.shape[3]) 
        val1 = (torch.exp(q_t_r)*(q_t_r - p_t - p_r_given_t)).view(b, -1).sum(1)  # 

        kl_z = kl_divergence(q_z_given_t_r, prior_z).sum(1) 
        kl_theta = kl_divergence(q_theta_given_t_r, prior_rot)
        val2 = (torch.exp(q_t_r) * (kl_theta + kl_z)).view(b, -1).sum(1)

        kl_div = val1 + val2



    size = y.size(1)
    log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y)*size
    elbo = log_p_x_g_z - kl_div

    return elbo, log_p_x_g_z, kl_div





def train_epoch(iterator, x_coord, generator_model, encoder_model, optim, translation_inference, rotation_inference
                , epoch, num_epochs, N, device, params, theta_prior):

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

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, generator_model, encoder_model, translation_inference, 
                            rotation_inference, epoch, device, theta_prior)

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
               , device, theta_prior):

    generator_model.eval()
    encoder_model.eval()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for y, in iterator:
        b = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, generator_model, encoder_model, 
                        translation_inference , rotation_inference, epoch, device, theta_prior)

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

    parser = argparse.ArgumentParser('Train Rotation Equivariant spatial-VAE on MNIST datasets')

    parser.add_argument('--dataset', choices=['mnist', 'mnist-rotated', 'mnist-rotated-translated-notCropped', 'mnist-rotated-translated'], default='mnist-rotated-translated', help='which MNIST datset to train/validate on (default: mnist-rotated-translated)')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--translation-inference', default='unimodal', choices=['unimodal', 'attention'], help='unimodal | attention')
    
    parser.add_argument('--rotation-inference', default='unimodal', choices=['unimodal', 'attention', 'attention+refinement'], help='unimodal | attention | attention+refinement')
    
    parser.add_argument('--groupconv', type=int, default=0, choices=[0, 4, 8, 16], help='0 | 4 | 8 | 16')
    parser.add_argument('--encoder-num-layers', type=int, default=2, help='number of hidden layers in original spatial-VAE inference model')
    parser.add_argument('--encoder-kernel-number', type=int, default=500, help='number of kernels in each layer of the encoder (default: 128)')

    parser.add_argument('--image-dim', type=int, default=28, help='input image of the shape image_dim x image_dim')
    parser.add_argument('--expand-coords', action='store_true', help='using random fourier feature expansion')


    parser.add_argument('--generator-hidden-dim', type=int, default=500, help='dimension of hidden layers (default: 500)')
    parser.add_argument('--generator-num-layers', type=int, default=2, help='number of hidden layers (default: 2)')
    parser.add_argument('--generator-resid-layers', action="store_true", help='using skip connections in generator')
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')

    parser.add_argument('--theta-prior', type=float, default=np.pi/4, help='standard deviation on rotation prior (default: pi/4)')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--log-root', default='./training_logs', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')

    args = parser.parse_args()
    num_epochs = args.num_epochs

    digits = int(np.log10(num_epochs)) + 1

    ## load the images
    if args.dataset == 'mnist':
        print('# training on MNIST', file=sys.stderr)
        mnist_train = torchvision.datasets.MNIST('data/mnist/', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST('data/mnist/', train=False, download=True)

        array = np.zeros((len(mnist_train),28,28), dtype=np.uint8)
        for i in range(len(mnist_train)):
            array[i] = np.array(mnist_train[i][0], copy=False)
        mnist_train = array

        array = np.zeros((len(mnist_test),28,28), dtype=np.uint8)
        for i in range(len(mnist_test)):
            array[i] = np.array(mnist_test[i][0], copy=False)
        mnist_test = array

    elif args.dataset == 'mnist-rotated':
        print('# training on rotated MNIST', file=sys.stderr)
        mnist_train = np.load('data/mnist_rotated/images_train.npy')
        mnist_test = np.load('data/mnist_rotated/images_test.npy')

    elif args.dataset == 'mnist-rotated-translated-notCropped':
        print('# training on rotated and translated (without cropping the digit) MNIST', file=sys.stderr)
        mnist_train = np.load('data/mnist_rotated_translated_notCropped/images_train.npy')
        mnist_test = np.load('data/mnist_rotated_translated_notCropped/images_test.npy')

    else:
        print('# training on rotated and translated MNIST', file=sys.stderr)
        mnist_train = np.load('data/mnist_rotated_translated/images_train.npy')
        mnist_test = np.load('data/mnist_rotated_translated/images_test.npy')

    mnist_train = torch.from_numpy(mnist_train).float()/255
    mnist_test = torch.from_numpy(mnist_test).float()/255

    n = args.image_dim

    ## x coordinate array
    xgrid = np.linspace(-1, 1, n)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    y_train = mnist_train.view(-1, n*n)
    y_test = mnist_test.view(-1, n*n)



    feature_expansion = args.expand_coords
    if feature_expansion:
        print('# Using random Fourier feature expansion', file=sys.stderr)


    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if use_cuda:
        #torch.cuda.set_device(d)
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

    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'leakyrelu':
        activation = nn.LeakyReLU


    # defining generator_model
    generator_model = models.SpatialGenerator(z_dim, generator_hidden_dim, num_layers=generator_num_layers
                                              , activation=activation, resid=generator_resid
                                              , expand_coords=feature_expansion)

    # defining encoder_model model
    translation_inference = args.translation_inference
    rotation_inference = args.rotation_inference
    encoder_num_layers = args.encoder_num_layers
    encoder_kernel_number = args.encoder_kernel_number
    group_conv = args.groupconv

    print('# translation inference is {}'.format(translation_inference), file=sys.stderr)
    print('# rotation inference is {}'.format(rotation_inference), file=sys.stderr)
    
    if translation_inference=='unimodal' and rotation_inference=='unimodal': #original spatial-vae from Bepler et. al 2019
        inf_dim = z_dim + 3 # 1 additional dim for rotation and 2 for translation 
        encoder_model = models.InferenceNetwork_UnimodalTranslation_UnimodalRotation(n*n, inf_dim, encoder_kernel_number, num_layers=encoder_num_layers, activation=activation)

    elif translation_inference=='attention' and rotation_inference=='unimodal':
        encoder_model = models.InferenceNetwork_AttentionTranslation_UnimodalRotation(n, z_dim, kernels_num=encode_kernel_number, activation=activation, groupconv=group_conv)

    elif translation_inference=='attention' and (rotation_inference=='attention' or rotation_inference=='attention+refinement'):
        rot_refinement = (rotation_inference=='attention+refinement')
        encoder_model = models.InferenceNetwork_AttentionTranslation_AttentionRotation(n, z_dim, kernels_num=encoder_kernel_number, activation=activation, groupconv=group_conv, rot_refinement=rot_refinement)


    generator_model.to(device)
    encoder_model.to(device)

    theta_prior = args.theta_prior

    print('# using priors: theta={}'.format(theta_prior), file=sys.stderr)
    print(encoder_model)

    N = len(mnist_train)

    params = list(generator_model.parameters()) + list(encoder_model.parameters())

    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)

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
                                      ,args.dataset, 'zDim', str(z_dim)])
    path_prefix = os.path.join(log_root, experiment_description,'')

    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)   



    save_interval = args.save_interval
    train_log = ['' for _ in range(2*num_epochs)]

    for epoch in range(num_epochs):
        elbo_accum,gen_loss_accum,kl_loss_accum = train_epoch(train_iterator, x_coord, generator_model, encoder_model, optim
                                                              , translation_inference=translation_inference
                                                              , rotation_inference=rotation_inference, epoch=epoch
                                                              , num_epochs=num_epochs, N=N, device=device, params=params
                                                              , theta_prior=theta_prior)

        line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        train_log[2*epoch] = line
        print(line, file=output)
        output.flush()

        # evaluate on the test set
        elbo_accum,gen_loss_accum,kl_loss_accum = eval_model(test_iterator, x_coord, generator_model, encoder_model
                                                             , translation_inference=translation_inference
                                                             , rotation_inference=rotation_inference, epoch=epoch
                                                             , device=device, theta_prior=theta_prior)
        
        line = '\t'.join([str(epoch+1), 'test', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        train_log[(2*epoch)+1] = line
        print(line, file=output)
        output.flush()

        #print('sigma is {}'.format(generator_model.embed_latent.sigma))
        ## save the models
        if path_prefix is not None and (epoch+1)%save_interval == 0:
            epoch_str = str(epoch+1).zfill(digits)

            path = path_prefix + '_generator_model_epoch{}.sav'.format(epoch_str)
            generator_model.eval().cpu()
            torch.save(generator_model, path)

            path = path_prefix + '_inference_epoch{}.sav'.format(epoch_str)
            encoder_model.eval().cpu()
            torch.save(encoder_model, path)

            generator_model.to(device)
            encoder_model.to(device)




if __name__ == '__main__':
    main()





