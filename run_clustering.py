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

from sklearn.cluster import KMeans
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
            probs, theta_vals, z_vals = encoder_model(y)
            
            #getting most probable t
            val, ind1 = probs.view(probs.shape[0], -1).max(1)
            ind0 = torch.arange(ind1.shape[0])
            
            #probs returned here is over the locations since rotation_inference is unimodal
            probs = probs.view(probs.shape[0], -1).unsqueeze(2)
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

            #btw_pixels_space = 0.0741
            x_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)
            y_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)[::-1]
            x_0,x_1 = np.meshgrid(x_grid, y_grid)
            x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
            x_coord_translate = torch.from_numpy(x_coord_translate).float().to(device)
            x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
            x_coord_translate = x_coord_translate.transpose(1, 2)
            dx = torch.bmm(x_coord_translate, probs).squeeze(2)

            # selecting theta_means from the most probable t
            theta_mu = theta_vals[ind0, 0:1, ind1]


    else:
        with torch.no_grad():
            attn, probs, theta_vals, z_vals = encoder_model(y, 100)
            
            #getting most probable t_r
            val, ind1 = probs.view(probs.shape[0], -1).max(1)
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
              
            probs_over_locs = torch.sum(probs, dim=1).view(probs.shape[0], -1, 1)
            
            #btw_pixels_space = 0.0741
            x_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)
            y_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)[::-1]
            x_0,x_1 = np.meshgrid(x_grid, y_grid)
            x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
            x_coord_translate = torch.from_numpy(x_coord_translate).to(device)
            x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))
            x_coord_translate = x_coord_translate.transpose(1, 2)
            dx = torch.bmm(x_coord_translate.type(torch.float), probs_over_locs).squeeze(2)
            
            # selecting theta_means from the most probable t_r
            theta_mu = theta_vals[ind0, 0:1, ind1]


    return z_content, theta_mu, dx 





def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    mapping = linear_sum_assignment(w.max() - w) # ind has the mapping from the true_labels to clusters

    sum_ = 0
    for i in range(len(mapping[0])):
        sum_ += w[mapping[0][i]][mapping[1][i]]

    return mapping, (sum_/y_pred.shape[0])



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

    parser = argparse.ArgumentParser('Train spatial-VAE on MNIST datasets')

    parser.add_argument('--dataset', choices=['mnist', 'mnist-rotated', 'mnist-rotated-translated-notCropped', 'mnist-rotated-translated'], default='mnist-rotated-translated', help='which MNIST datset to train/validate on (default: mnist-rotated-translated)')
    parser.add_argument('--path-to-model', help='path to the saved encoder model')
    parser.add_argument('--path-to-test-file', default='./data/mnist/MNIST/processed/test.pt', help='path to the file that has labels of the test images')

    parser.add_argument('--path-to-transformations', default='./data/mnist_rotated_translated/transforms_test.npy', help='path to the file that has the ground-truth values for the translation and rotation')


    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--translation-inference', default='unimodal', choices=['unimodal', 'attention'], help='unimodal | attention')
    parser.add_argument('--rotation-inference', default='unimodal', choices=['unimodal', 'attention', 'attention+refinement'], help='rotation+refinement can only be done when using the group-conv layers')

    parser.add_argument('--groupconv', type=int, default=0, choices=[0, 4, 8, 16], help='0 | 4 | 8 | 16')
    parser.add_argument('--encoder-num-layers', type=int, default=2, help='number of hidden layers in original spatial-VAE inference model')
    parser.add_argument('--encoder-kernel-number', type=int, default=500, help='number of kernels in each layer of the encoder (default: 128)')

    parser.add_argument('--image-dim', type=int, default=28, help='input image of the shape image_dim x image_dim')
    parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')

    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')

    args = parser.parse_args()

    ## load the images; Other datasets can be added by increasing the choices
    if args.dataset == 'mnist':
        print('# training on MNIST', file=sys.stderr)
        train_set = torchvision.datasets.MNIST('data/mnist/', train=True, download=True)
        test_set = torchvision.datasets.MNIST('data/mnist/', train=False, download=True)

        array = np.zeros((len(train_set),28,28), dtype=np.uint8)
        for i in range(len(train_set)):
            array[i] = np.array(train_set[i][0], copy=False)
        train_set = array

        array = np.zeros((len(test_set),28,28), dtype=np.uint8)
        for i in range(len(test_set)):
            array[i] = np.array(test_set[i][0], copy=False)
        test_set = array

    elif args.dataset == 'mnist-rotated':
        print('# training on rotated MNIST', file=sys.stderr)
        train_set = np.load('data/mnist_rotated/images_train.npy')
        test_set = np.load('data/mnist_rotated/images_test.npy')

    elif args.dataset == 'mnist-rotated-translated-notCropped':
        print('# training on rotated and translated (without cropping the digit) MNIST', file=sys.stderr)
        train_set = np.load('data/mnist_rotated_translated_notCropped/images_train.npy')
        test_set = np.load('data/mnist_rotated_translated_notCropped/images_test.npy')

    else:
        print('# training on rotated and translated MNIST', file=sys.stderr)
        train_set = np.load('data/mnist_rotated_translated/images_train.npy')
        test_set = np.load('data/mnist_rotated_translated/images_test.npy')

        

    train_set = torch.from_numpy(train_set).float()/255
    test_set = torch.from_numpy(test_set).float()/255
    mnist_test = torch.load(args.path_to_test_file)
    y_labels = mnist_test[1]

    n = args.image_dim

    ## x coordinate array
    xgrid = np.linspace(-1, 1, n)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    y_train = train_set.view(-1, n*n)
    y_test = test_set.view(-1, n*n)

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


    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'leakyrelu':
        activation = nn.LeakyReLU


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
        encoder_model = models.InferenceNetwork_AttentionTranslation_UnimodalRotation(n, z_dim, kernels_num=encoder_kernel_number, activation=activation, groupconv=group_conv)

    elif translation_inference=='attention' and (rotation_inference=='attention' or rotation_inference=='attention+refinement'):
        rot_refinement = (rotation_inference=='attention+refinement')
        encoder_model = models.InferenceNetwork_AttentionTranslation_AttentionRotation(n, z_dim, kernels_num=encoder_kernel_number, activation=activation, groupconv=group_conv, rot_refinement=rot_refinement)


    encoder_model.to(device)
    path_to_model = args.path_to_model
    encoder = torch.load(path_to_model).to(device)

    minibatch_size = args.minibatch_size
    test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)

    #folder for writing log files
    path_prefix = '/'.join(path_to_model.split('/')[:-1]) 

    z_values = torch.empty(len(data_test), 2*z_dim)
    tr_pred = torch.empty(len(data_test), 2)
    rot_pred = torch.empty(len(data_test), 1)

    for i in range(0,len(data_test), minibatch_size):
        y = data_test[i:i+minibatch_size]
        y = torch.stack(y, dim=0).squeeze(0).to(device)

        a, b, c = get_latent(x_coord, y, encoder, translation_inference, rotation_inference, 'cuda:0')

        z_values[i:i+minibatch_size] = a.cpu()
        rot_pred[i:i+minibatch_size] = b.cpu()
        tr_pred[i:i+minibatch_size]  = c.cpu()




    cluster_model = KMeans(10).fit(z_values.detach())
    cluster = cluster_model.predict(z_values.detach())

    tsne = TSNE(2).fit_transform(z_values.detach())


    # accuracy of clustering
    print('calculating clustering accuracy ... ', file=sys.stderr)
    mapping, acc = cluster_acc(y_labels.cpu().numpy(), cluster)


    #calculate translation correlation and rotation correlation
    if args.dataset != 'mnist':
        print('calculating the correlation for the rotation and translation ... ', file=sys.stderr)
        rot_corr, tr_corr = measure_correlations(args.path_to_transformations, rot_pred, tr_pred)


    # saving confusion matrix as a figure
    print('saving confusion matrix ... ', file=sys.stderr)
    cm = confusion_matrix(y_labels, cluster)
    sns.set()
    ax = sns.heatmap(cm[:, np.array(mapping[1])], annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10))
    ax=ax.set(xlabel='clusters', ylabel='true_labels')
    plt.savefig(path_prefix + "/confusion_matrix.jpg")


    # saving tsne figure
    print('saving tsne figure ... ', file=sys.stderr)
    plt.figure(figsize=(10, 10))

    cmap = plt.cm.rainbow
    #cmap = matplotlib.colors.ListedColormap(['red', 'cyan','yellow', 'orange', 'green', 'purple',
    #            'black', 'darkblue','darkcyan', 'gold'])
    #cmap = cm.get_cmap('viridis', 10)
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



    with open(path_prefix + '/results.txt', 'w') as f:
        f.write('using the encoder model from {}\n\n'.format(path_to_model))
        f.write('The accuracy for clustering is {} \n'.format(acc))
        f.write('The circular correlation for the rotation is {}\n'.format(rot_corr))
        f.write('The Pearson correlation for the x and y values in the translation is {}\n'.format(tr_corr))





if __name__ == '__main__':
    main()


