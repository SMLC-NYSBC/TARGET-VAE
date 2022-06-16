from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys
import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from  matplotlib import cm
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from astropy.stats import circcorrcoef
from astropy import units as u

import src.models as models


def load_images(path):
    if path.endswith('mrc') or path.endswith('mrcs'):
        with open(path, 'rb') as f:
            content = f.read()
        images,_,_ = mrc.parse(content)
    elif path.endswith('npy'):
        images = np.load(path)
    return images


 

def get_latent(x, y, encoder_model, translation_inference, rotation_inference, device):

    b = y.size(0)
    btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()
    x = x.expand(b, x.size(0), x.size(1)).to(device)
    y = y.to(device)
    
    if translation_inference == 'unimodal' and rotation_inference == 'unimodal':
        with torch.no_grad():
            y = y.view(b, -1)
            z_mu,z_logstd = encoder_model(y)
            z_std = torch.exp(z_logstd)
            z_dim = z_mu.size(1)

            # z[0] is the rotation
            theta_mu = z_mu[:,0].unsqueeze(1)

            dx_mu = z_mu[:,1:3]
            dx = dx_mu

            z_content = torch.cat((z_mu[:,3:], z_std[:,3:]), dim=1)


    elif translation_inference == 'attention' and rotation_inference == 'unimodal':
        with torch.no_grad():
            attn, sampled_attn, theta_vals, z_vals = encoder_model(y)
            
            #getting most probable t
            val, ind1 = attn.view(attn.shape[0], -1).max(1)
            ind0 = torch.arange(ind1.shape[0])
            
            z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
            theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)

            z_dim = z_vals.size(1) // 2
            z_mu = z_vals[:,:z_dim, ]
            z_logstd = z_vals[:, z_dim:, ]
            z_std = torch.exp(z_logstd)

            # selecting z_values from the most probable t
            z_mu = z_mu[ind0, :, ind1]
            z_std = z_std[ind0, :, ind1]
            z_content = torch.cat((z_mu, z_std), dim=1)
            
            attn_softmax = F.softmax(attn.view(b, -1), dim=1).unsqueeze(2)
            
            attn_dim = attn.shape[3]
            if  attn_dim % 2:
                x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)[::-1]
            else:
                x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)[::-1]
            x_0,x_1 = np.meshgrid(x_grid, y_grid)
            x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
            x_coord_translate = torch.from_numpy(x_coord_translate).float().to(device)
            x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
            x_coord_translate = x_coord_translate.transpose(1, 2)
            dx = torch.bmm(x_coord_translate, attn_softmax).squeeze(2)

            # selecting theta_means from the most probable t
            theta_mu = theta_vals[ind0, 0:1, ind1]
            

    else:
        with torch.no_grad():
            attn, _, _, _, _, theta_vals, z_vals = encoder_model(y)
            
            #getting most probable t_r
            val, ind1 = attn.view(attn.shape[0], -1).max(1)
            ind0 = torch.arange(ind1.shape[0])
            
            z_vals = z_vals.view(z_vals.shape[0], z_vals.shape[1], -1)
            theta_vals = theta_vals.view(theta_vals.shape[0], theta_vals.shape[1], -1)
            

            z_dim = z_vals.size(1) // 2
            z_mu = z_vals[:,:z_dim, ]
            z_logstd = z_vals[:, z_dim:, ]
            z_std = torch.exp(z_logstd) 
            
            # selecting z_values from the most probable t_r
            z_mu = z_mu[ind0, :, ind1]
            z_std = z_std[ind0, :, ind1]
            z_content = torch.cat((z_mu, z_std), dim=1)
              
            attn_softmax = F.softmax(attn.view(b, -1), dim=1).view(attn.shape).sum(1).view(b, -1).unsqueeze(2)
            
            attn_dim = attn.shape[3]
            if  attn_dim % 2:
                x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2 + 1), btw_pixels_space)[::-1]
            else:
                x_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)
                y_grid = np.arange(-btw_pixels_space*(attn_dim//2), btw_pixels_space*(attn_dim//2), btw_pixels_space)[::-1]
            x_0,x_1 = np.meshgrid(x_grid, y_grid)
            x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
            x_coord_translate = torch.from_numpy(x_coord_translate).to(device)
            x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
            x_coord_translate = x_coord_translate.transpose(1, 2)
            dx = torch.bmm(x_coord_translate.type(torch.float), attn_softmax).squeeze(2)
            
            # selecting theta_means from the most probable t_r
            theta_mu = theta_vals[ind0, 0:1, ind1]


    return z_content, theta_mu, dx 






def measure_correlations(path_to_transformations, rot_pred, tr_pred):
    test_transforms = np.load(path_to_transformations)
    rot_val = test_transforms[:, 0].reshape(test_transforms.shape[0], 1)
    tr_val = test_transforms[:, 1:].reshape(test_transforms.shape[0], 2)

    rot_corr = circcorrcoef(rot_val, rot_pred.numpy())
    x_corr = np.corrcoef(tr_val[:,0], tr_pred.numpy()[:,0])[0][1]
    y_corr = np.corrcoef(tr_val[:,1], tr_pred.numpy()[:,1])[0][1]
    tr_corr = [x_corr, y_corr]

    return rot_corr, tr_corr




def main():
    import argparse

    parser = argparse.ArgumentParser('Clustering particles...')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--test-path', help='path to the whole data; or path to testing data')
    parser.add_argument('--path-to-encoder', help='path to the saved encoder model')
    
    parser.add_argument('--path-to-transformations', help='path to a single file that contains the ground-truth rotation in the first column, and the ground-truth translation values in the second and third columns; This is required for calculating rotation and translation correlation between the predicted values and the ground-truth ones')

    parser.add_argument('--t-inf', default='unimodal', choices=['unimodal', 'attention'], help='unimodal | attention')
    parser.add_argument('--r-inf', default='unimodal', choices=['unimodal', 'attention', 'attention+offsets'], help='unimodal | attention | attention+offsets')
    
    parser.add_argument('--clustering', default='agglomerative', choices=['agglomerative', 'k-means'], help='agglomerative | k-means')
    parser.add_argument('--normalize', action='store_true', help='normalize the images before training')
    parser.add_argument('--crop', default=0, type=int, help='size of the cropped images')
    
    parser.add_argument('--in-channels', type=int, default=1, help='number of channels in the images')
    
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')
    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')

    args = parser.parse_args()

    ## load the images
    images_test = load_images(args.test_path)
    
    crop = args.crop
    if crop > 0:
        images_test = image_utils.crop(images_test, crop)
        print('# cropped to:', crop, file=sys.stderr)
    
    n,m = images_test.shape[1:]
    
    # normalize the images using edges to estimate background
    if args.normalize:
        print('# normalizing particles', file=sys.stderr)
        mu = images_test.reshape(-1, m*n).mean(1)
        std = images_test.reshape(-1, m*n).std(1)
        images_test = (images_test - mu[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
    
    # x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()
    
    images_test = torch.from_numpy(images_test).float()
    in_channels = 1
    y_test = images_test.view(-1, in_channels, n, m)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(d)
        print('# using CUDA device:', d, file=sys.stderr)
        device = torch.device("cuda:" + str(d) if use_cuda else "cpu")
    else:
        device = torch.device("cpu")

    y_test = y_test.to(device)
    x_coord = x_coord.to(device)

    data_test = torch.utils.data.TensorDataset(y_test)

    z_dim = args.z_dim
    print('# clustering with z-dim:', z_dim, file=sys.stderr)

    # defining encoder model
    translation_inference = args.t_inf
    rotation_inference = args.r_inf

    print('# translation inference is {}'.format(translation_inference), file=sys.stderr)
    print('# rotation inference is {}'.format(rotation_inference), file=sys.stderr)

    path_to_encoder = args.path_to_encoder
    encoder = torch.load(path_to_encoder).to(device)

    minibatch_size = args.minibatch_size

    #folder for writing log files
    path_prefix = '/'.join(path_to_encoder.split('/')[:-1]) 

    z_values = torch.empty(len(data_test), 2*z_dim)
    tr_pred = torch.empty(len(data_test), 2)
    rot_pred = torch.empty(len(data_test), 1)
    
    # getting predicted z, rotation, and translation for the data
    for i in range(0,len(data_test), minibatch_size):
        y = data_test[i:i+minibatch_size]
        y = torch.stack(y, dim=0).squeeze(0).to(device)

        a, b, c = get_latent(x_coord, y, encoder, translation_inference, rotation_inference, device)

        z_values[i:i+minibatch_size] = a.cpu()
        rot_pred[i:i+minibatch_size] = b.cpu()
        tr_pred[i:i+minibatch_size]  = c.cpu()

    
    if args.clustering == 'agglomerative':
        # AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=10, linkage='ward', compute_full_tree=True)
        cluster = ac.fit_predict(z_values.detach().cpu())
    elif args.clustering == 'k-means':
        # k-means clustering
        km = KMeans(10, n_init=100).fit(z_values.detach().cpu())
        cluster = km.predict(z_values.detach().cpu())

    if args.path_to_transformations:
        rot_corr, tr_corr = measure_correlations(args.path_to_transformations, rot_pred, tr_pred)
    
    
    '''
    # saving tsne figure
    print('# saving tsne figure ... ', file=sys.stderr)
    tsne = TSNE(2, learning_rate=200.0, init='random').fit_transform(z_values.detach())
    plt.figure(figsize=(10, 10))

    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(0, 11, 1), cmap.N)

    plt.scatter(tsne[:, 0], tsne[:, 1], cmap=cmap, norm=norm, s=2)

    # to modify size of the colorbar
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)

    # to make the number on the colorbar centered
    cb = plt.colorbar(cax=cax)
    labels = np.arange(0, 10, 1)
    loc = labels + .5
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)

    plt.savefig(path_prefix + "/tsne.jpg")
    '''
    
    # saving histogram of predicted rotation values
    print('# saving chart of predicted rotation values ... ', file=sys.stderr)
    ax = plt.hist(rot_pred.detach().cpu().numpy())
    plt.xlabel('predicted rotation angles')
    plt.ylabel('samples')
    plt.savefig(path_prefix + "/predicted_rotation_vals.jpg")

    # saving histogram of predicted translation values
    print('# saving chart of predicted translation values ... ', file=sys.stderr)
    plt.hist(tr_pred[:, 0].detach().cpu().numpy())
    plt.xlabel('predicted translation values for x')
    plt.ylabel('samples')
    plt.savefig(path_prefix + "/predicted_translation_x_vals.jpg")
    
    plt.hist(tr_pred[:, 1].detach().cpu().numpy())
    plt.xlabel('predicted translation values for y')
    plt.ylabel('samples')
    plt.savefig(path_prefix + "/predicted_translation_y_vals.jpg")


    with open(path_prefix + '/results.txt', 'w') as f:
        f.write('using the encoder model from {}\n\n'.format(path_to_encoder))
        
        if args.path_to_transformations:
            f.write('The circular correlation for the rotation is {}\n'.format(rot_corr))
            f.write('The Pearson correlation for the x and y values in the translation is {}\n'.format(tr_corr))
        
        




if __name__ == '__main__':
    main()

