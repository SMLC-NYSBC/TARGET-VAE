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


 

def get_latent(x, y, encoder_model, t_inf, r_inf, device):
    """
    Arguments
        x: base coordinates of the pixels, not rotated or translated
        y: input 
        encoder_model: the encoder model
        t_inf: translation inference which can be 'unimodal' or 'attention'
        r_inf: rotation inference which can be 'unimodal' or 'attention' or 'attention+offsets'
        device: int
    Return
        z_content: rotation-translation-invariant representations
        theta_mu: predicted rotation for the object
        dx: prdicted translation for the object
    """
    b = y.size(0)
    btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()
    x = x.expand(b, x.size(0), x.size(1)).to(device)
    y = y.to(device)
    
    if t_inf == 'unimodal' and r_inf == 'unimodal':
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


    elif t_inf == 'attention' and r_inf == 'unimodal':
        with torch.no_grad():
            attn, sampled_attn, theta_vals, z_vals = encoder_model(y, device)
            
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
            

    else: # t_inf='attention' and r_inf='attention+offsets'
        with torch.no_grad():
            attn, _, _, _, _, theta_vals, z_vals = encoder_model(y, device)
            
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





def cluster_acc(y_true, y_pred):
    """
    Arguments
        y_true: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    Return
        mapping: mapping from the true_labels to the clusters
        accuracy of clustering
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    mapping = linear_sum_assignment(w.max() - w) 

    sum_ = 0
    for i in range(len(mapping[0])):
        sum_ += w[mapping[0][i]][mapping[1][i]]

    return mapping, (sum_/y_pred.shape[0])



def measure_correlations(r_gt, t_gt, r_pred, t_pred):
    """
    Arguments
        r_gt: ground-truth rotation angles
        t_gt: ground-truth translation values
        r_pred:predicted rotation angles
        t_pred: predicted translation values
    Return
        r_corr: circular rotatation correlation
        t_corr: Pearson correaltion coefficient for translations over x and y
    """
    r_corr = circcorrcoef(r_gt, r_pred.numpy())
    x_corr = np.corrcoef(t_gt[:,0], t_pred.numpy()[:,0])[0][1]
    y_corr = np.corrcoef(t_gt[:,1], t_pred.numpy()[:,1])[0][1]
    tr_corr = [x_corr, y_corr]

    return r_corr, t_corr




def main():
    import argparse

    parser = argparse.ArgumentParser('Clustering dSprites')

    parser.add_argument('--train-path', default='data/dsprites-dataset-master/imgs_train.npy', help='path to training data; or path to the whole data (default:data/dsprites-dataset-master/imgs_train.npy)')
    parser.add_argument('--test-path', default='data/dsprites-dataset-master/imgs_test.npy', help='path to testing data (default:data/dsprites-dataset-master/imgs_test.npy)')
    
    parser.add_argument('--train-labels', default='./data/dsprites-dataset-master/latent_train.npy', help='path to training data; or path to the whole data (default:./data/dsprites-dataset-master/latent_train.npy)')
    parser.add_argument('--test-labels', default='./data/dsprites-dataset-master/latent_test.npy' , help='path to testing data (default:./data/dsprites-dataset-master/latent_test.npy)')
    
    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--inp-channel', type=int, default=1, help='number of the channels in the input (default: 1)')
    
    parser.add_argument('--path-to-encoder', help='path to the saved encoder model')

    parser.add_argument('--t-inf', default='attention', choices=['unimodal', 'attention'], help='unimodal | attention')
    parser.add_argument('--r-inf', default='attention+offsets', choices=['unimodal', 'attention', 'attention+offsets']
                        , help='unimodal | attention | attention+offsets')
    
    parser.add_argument('--clustering', default='agglomerative', choices=['agglomerative', 'k-means'], help='agglomerative | k-means')
    parser.add_argument('--n-clusters', default=10, type=int, help='Number of clusters (default:10)')
    
    parser.add_argument('--in-channels', type=int, default=1, help='number of channels in the images')
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')

    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')
    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')

    args = parser.parse_args()

    ## load the images
    images_train = np.load(args.train_path)
    images_test = np.load(args.test_path) 
    images = np.concatenate((images_train, images_test))
    images = torch.from_numpy(images).float()
    print('**')
    print(images.shape)
    
    train_labels = np.load(args.train_labels)
    test_labels = np.load(args.test_labels)

    labels = np.concatenate((train_labels, test_labels))
    shape_labels = labels[:, 1]
    r_gt = labels[:, 3:4] # ground-truth rotation values
    t_gt = labels[:, 4: ] # ground-truth translation values

    n,m = images.shape[1:]
    
    ## x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()
    
    in_channels = args.in_channels
    y_test = images.view(-1, in_channels, n, m)

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
    t_inf = args.t_inf
    r_inf = args.r_inf

    print('# translation inference is {}'.format(t_inf), file=sys.stderr)
    print('# rotation inference is {}'.format(r_inf), file=sys.stderr)

    path_to_encoder = args.path_to_encoder
    encoder = torch.load(path_to_encoder).to(device)

    minibatch_size = args.minibatch_size

    #folder for writing log files
    path_prefix = '/'.join(path_to_encoder.split('/')[:-1]) 

    z_values = torch.empty(len(data_test), 2*z_dim)
    t_pred = torch.empty(len(data_test), 2)
    r_pred = torch.empty(len(data_test), 1)
    
    # getting predicted z, rotation, and translation for the data
    for i in range(0,len(data_test), minibatch_size):
        y = data_test[i:i+minibatch_size]
        y = torch.stack(y, dim=0).squeeze(0).to(device)
        a, b, c = get_latent(x_coord, y, encoder, t_inf, r_inf, device)

        z_values[i:i+minibatch_size] = a.cpu()
        r_pred[i:i+minibatch_size] = b.cpu()
        t_pred[i:i+minibatch_size]  = c.cpu()

    r_corr, t_corr = measure_correlations(r_gt, t_gt, r_pred, t_pred)
    
    n_clusters = args.n_clusters
    if args.clustering == 'agglomerative':
        # AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_full_tree=True)
        cluster = ac.fit_predict(z_values.detach().cpu())
    elif args.clustering == 'k-means':
        # k-means clustering
        km = KMeans(n_clusters=n_clusters, n_init=100).fit(z_values.detach().cpu())
        cluster = km.predict(z_values.detach().cpu())

    mapping, acc = cluster_acc(y_labels.cpu().numpy(), cluster)

    
    
    
    # saving tsne figure
    print('# saving tsne figure ... ', file=sys.stderr)
    tsne = TSNE(2, learning_rate=200.0, init='random').fit_transform(z_values.detach())
    plt.figure(figsize=(10, 10))

    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(0, 11, 1), cmap.N)

    plt.scatter(tsne[:, 0], tsne[:, 1], c=y_labels, cmap=cmap, norm=norm, s=2)

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
    
    
    
    
    # saving confusion matrix as a figure
    print('# saving confusion matrix ... ', file=sys.stderr)
    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_labels, cluster)
    sns.set()
    ax = sns.heatmap(cm[:, np.array(mapping[1])], annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10))
    ax=ax.set(xlabel='clusters', ylabel='true_labels')
    plt.savefig(path_prefix + "/confusion_matrix.jpg")


    



    with open(path_prefix + '/results.txt', 'w') as f:
        f.write('using the encoder model from {}\n\n'.format(path_to_encoder))
        f.write('The accuracy for clustering is {} \n'.format(acc))
        f.write('The circular correlation for the rotation is {}\n'.format(r_corr))
        f.write('The Pearson correlation for the x and y values in the translation is {}\n'.format(t_corr))





if __name__ == '__main__':
    main()

