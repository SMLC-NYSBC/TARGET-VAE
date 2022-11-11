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


 

def get_latent(x, y, encoder_model, t_inf, r_inf, device, image_dim):
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
            

    else:
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
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    Return
        accuracy
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



def measure_correlations(path_to_transformations, r_pred, t_pred):
    """
    Arguments
        path_to_transformation: path to the transformations file
        r_pred:predicted rotation angles
        t_pred: predicted translation values
    Return
        r_corr: circular rotatation correlation
        t_corr: Pearson correaltion coefficient for translations over x and y
    """
    test_transforms = np.load(path_to_transformations)
    rot_val = test_transforms[:, 0].reshape(test_transforms.shape[0], 1)
    t_val = test_transforms[:, 1:].reshape(test_transforms.shape[0], 2)

    r_corr = circcorrcoef(rot_val, r_pred.numpy())
    x_corr = np.corrcoef(t_val[:,0], t_pred.numpy()[:,0])[0][1]
    y_corr = np.corrcoef(t_val[:,1], t_pred.numpy()[:,1])[0][1]
    t_corr = [x_corr, y_corr]

    return r_corr, t_corr




def main():
    import argparse

    parser = argparse.ArgumentParser('Clustering mnist/mnist-N/mnist-U')

    parser.add_argument('--dataset', choices=['mnist', 'mnist-U', 'mnist-N'], default='mnist-U', help='which MNIST datset to train/validate on (default:mnist-U)')
    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default:2)')
    
    parser.add_argument('--path-to-encoder', help='path to the saved encoder model')
    parser.add_argument('--path-to-mnist-test', default='./data/MNIST/processed/test.pt', help='path to the file that has labels of the test images (default:./data/MNIST/processed/test.pt)')

    parser.add_argument('--t-inf', default='attention', choices=['unimodal', 'attention'], help='unimodal | attention (default:attention)')
    parser.add_argument('--r-inf', default='attention+offsets', choices=['unimodal', 'attention', 'attention+offsets'], help='unimodal | attention | attention+offsets (default:attention+offsets)')
    
    parser.add_argument('--clustering', default='agglomerative', choices=['agglomerative', 'k-means'], help='agglomerative | k-means (default:agglomerative)')
    parser.add_argument('--n-clusters', default=10, type=int, help='Number of clusters (default:10)')
    
    parser.add_argument('--in-channels', type=int, default=1, help='number of channels in the images (default:1)')
    parser.add_argument('--image-dim', type=int, default=50, help='input image of the shape image_dim x image_dim (default:50)')
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')

    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default:100)')
    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use (default:0)')

    args = parser.parse_args()

    ## load the images
    if args.dataset == 'mnist':
        mnist_test = torchvision.datasets.MNIST('data/', train=False, download=True)

        array = np.zeros((len(mnist_test), args.image_dim, args.image_dim), dtype=np.uint8)
        for i in range(len(mnist_test)):
            array[i] = np.array(mnist_test[i][0], copy=False)
        mnist_test = array
        
        path_to_transformations = None #no transformation on standard MNIST

    elif args.dataset == 'mnist-U':
        mnist_test = np.load('data/mnist_U/images_test.npy')
        path_to_transformations = 'data/mnist_U/transforms_test.npy'
    
    elif args.dataset == 'mnist-N':
        mnist_test = np.load('data/mnist_N/images_test.npy')
        path_to_transformations = 'data/mnist_N/transforms_test.npy'
        
    else:
        print('# Wrong value for the dataset!', file=sys.stderr)
        return
    
    mnist_test = torch.from_numpy(mnist_test).float()/255
    y_labels = torch.load(args.path_to_mnist_test)[1]

    image_dim = args.image_dim

    ## x coordinate array
    xgrid = np.linspace(-1, 1, image_dim)
    ygrid = np.linspace(1, -1, image_dim)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()
    
    in_channels = 1
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
    tr_pred = torch.empty(len(data_test), 2)
    rot_pred = torch.empty(len(data_test), 1)
    
    # getting predicted z, rotation, and translation for the transformed mnist_N or mnist_U datasets
    for i in range(0,len(data_test), minibatch_size):
        y = data_test[i:i+minibatch_size]
        y = torch.stack(y, dim=0).squeeze(0).to(device)

        a, b, c = get_latent(x_coord, y, encoder, t_inf, r_inf, device, image_dim)

        z_values[i:i+minibatch_size] = a.cpu()
        rot_pred[i:i+minibatch_size] = b.cpu()
        tr_pred[i:i+minibatch_size]  = c.cpu()

        
    # To calculate the predicted rotation and translation values for mnist_N and mnist_U and measure the correlation
    # , we need to measure these values fro the digits in the regular mnist first, because of the slight rotation and
    #  translation of the digits in the mnist dataset. Then we use (pred_on_transformed_data - pred_on_regular_mnist)
    # to calculate the correlations
    if args.dataset != 'mnist':
        print('# calculating the correlation for the rotation and translation ... ', file=sys.stderr)
        mnist_test = torch.load(args.path_to_mnist_test)[0]/255
        m = nn.ZeroPad2d((image_dim - mnist_test[0].shape[1])//2)
        mnist_test = m(mnist_test)
        mnist_test = mnist_test.view(-1, 1, image_dim, image_dim)
        mnist_test = torch.utils.data.TensorDataset(mnist_test)
        
        tr_pred_mnist = torch.empty(len(mnist_test), 2)
        rot_pred_mnist = torch.empty(len(mnist_test), 1)
        for i in range(0,len(mnist_test), minibatch_size):
            y = mnist_test[i:i+minibatch_size]
            y = torch.stack(y, dim=0).squeeze(0).to(device)

            _, b, c = get_latent(x_coord, y, encoder, t_inf, r_inf, device, image_dim)

            rot_pred_mnist[i:i+minibatch_size] = b.cpu()
            tr_pred_mnist[i:i+minibatch_size]  = c.cpu()
        
        rot_corr, tr_corr = measure_correlations(path_to_transformations, rot_pred - rot_pred_mnist, tr_pred - tr_pred_mnist)
    
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
        f.write('The circular correlation for the rotation is {}\n'.format(rot_corr))
        f.write('The Pearson correlation for the x and y values in the translation is {}\n'.format(tr_corr))





if __name__ == '__main__':
    main()

