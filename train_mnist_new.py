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




def eval_minibatch(epoch, x, y, generator_model, encoder_model, rotate=True, translate=True,
				   use_cuda=False):
	
	b = y.size(0)
	btw_pixels_space = (x[1, 0] - x[0, 0]).cpu().numpy()	
	x = x.expand(b, x.size(0), x.size(1)).cuda()

	if use_cuda:
		y = y.cuda()
	
	
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
	r_z = rand_dist.sample((b, z_dim)).cuda()
	z = (z_std_expected*r_z + z_mu_expected).squeeze(2)
 
	if translate:
		#btw_pixels_space = 0.0741
		x_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)
		y_grid = np.arange(-btw_pixels_space*27, btw_pixels_space*28, btw_pixels_space)[::-1]
		x_0,x_1 = np.meshgrid(x_grid, y_grid)
		x_coord_translate = np.stack([x_0.ravel(), x_1.ravel()], 1)
		x_coord_translate = torch.from_numpy(x_coord_translate).cuda()
		x_coord_translate = x_coord_translate.expand(b, x_coord_translate.size(0), x_coord_translate.size(1))

		x_coord_translate = x_coord_translate.transpose(1, 2)

		dx_expected = torch.bmm(x_coord_translate.type(torch.float), probs_over_locs).squeeze(2).unsqueeze(1)
		x = x - dx_expected # translate coordinates
		
	
	kl_div = 0
	if rotate:
		theta_mu = theta_vals[:, 0:1, ]
		theta_logstd = theta_vals[:, 1:2, ]
		theta_std = torch.exp(theta_logstd) 
		
		theta_mu_expected = torch.bmm(theta_mu, probs)
		theta_std_expected = torch.bmm(theta_std, probs)
		
		r_theta = rand_dist.sample((b, 1)).cuda()
		theta = (theta_std_expected*r_theta + theta_mu_expected).squeeze(2).squeeze(1) 
		
		
		# calculate rotation matrix
		rot = Variable(theta.data.new(b,2,2).zero_())
		rot[:,0,0] = torch.cos(theta)
		rot[:,0,1] = torch.sin(theta)
		rot[:,1,0] = -torch.sin(theta)
		rot[:,1,1] = torch.cos(theta)
		x = torch.bmm(x, rot) # rotate coordinates by theta
		
	
	
	# reconstruct
	y_hat = generator_model(x, z).squeeze(2)
	size = y.size(1)
	log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y)*size
	
	
	
	q_z_given_t_r = Normal(z_mu.view(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3]), z_std.view(b, z_dim, attn.shape[1], attn.shape[2], attn.shape[3])) # B x z_dim x R x HW
	q_theta_given_t_r = Normal(theta_mu.view(b, attn.shape[1], attn.shape[2], attn.shape[3]), theta_std.view(b, attn.shape[1], attn.shape[2], attn.shape[3]))
	
	q_t_r = F.log_softmax(attn.view(b, -1), dim=1).view(b, attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W
	
	# normal prior over r
	p_r_given_t = torch.zeros_like(q_t_r).cuda() - np.log(attn.shape[1])
	
	
	# uniform prior over t
	#p_t = torch.zeros_like(q_t_r).cuda() - np.log(attn.shape[2]*attn.shape[3])
	
	
	# normal prior over t
	p_t = torch.zeros(1, 1, attn.shape[2], attn.shape[3]).cuda()
	p_t[:, :, :, :] = p_t_dist.log_prob(torch.tensor([x_grid]).cuda()).transpose(0, 1) + p_t_dist.log_prob(torch.tensor([y_grid]).cuda())
	p_t = p_t.expand(b, attn.shape[1], p_t.shape[2], p_t.shape[3])

	

	val1 = (torch.exp(q_t_r)*(q_t_r - p_t - p_r_given_t)).view(b, -1).sum(1)
	
	kl_z = kl_divergence(q_z_given_t_r, prior_z).sum(1) 
	kl_theta = kl_divergence(q_theta_given_t_r, prior_rot)
	
	val2 = (torch.exp(q_t_r) * (kl_theta + kl_z)).view(b, -1).sum(1)
	
	kl_div = val1 + val2
	kl_div = kl_div.mean() 

	elbo = log_p_x_g_z - kl_div

	return elbo, log_p_x_g_z, kl_div





def train_epoch(iterator, x_coord, generator_model, encoder_model, optim, rotate=True, translate=True
				, epoch=1, num_epochs=1, N=1, use_cuda=False, params=None):
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

		elbo, log_p_x_g_z, kl_div = eval_minibatch(epoch, x, y, generator_model, encoder_model, rotate=rotate
												   , translate=translate, use_cuda=use_cuda)
		
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





def eval_model(epoch, iterator, x_coord, generator_model, encoder_model, rotate=True, translate=True
			  , use_cuda=False):
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

		elbo, log_p_x_g_z, kl_div = eval_minibatch(epoch, x, y, generator_model, encoder_model
												   , rotate=rotate, translate=translate, use_cuda=use_cuda)

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





dataset = 'mnist-rotated-translated'

## load the images
if dataset == 'mnist':
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

elif dataset == 'mnist-rotated':
	print('# training on rotated MNIST', file=sys.stderr)
	mnist_train = np.load('data/mnist_rotated/images_train.npy')
	mnist_test = np.load('data/mnist_rotated/images_test.npy')

else:
	print('# training on rotated and translated MNIST', file=sys.stderr)
	mnist_train = np.load('data/mnist_rotated_translated/images_train.npy')
	mnist_test = np.load('data/mnist_rotated_translated/images_test.npy')


# In[6]:


mnist_train = torch.from_numpy(mnist_train).float()/255
mnist_test = torch.from_numpy(mnist_test).float()/255

n = m = 28


# In[7]:


## x coordinate array
xgrid = np.linspace(-1, 1, m)
ygrid = np.linspace(1, -1, n)
x0,x1 = np.meshgrid(xgrid, ygrid)
x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
x_coord = torch.from_numpy(x_coord).float()

y_train = mnist_train.view(-1, n*m)
y_test = mnist_test.view(-1, n*m)


# In[8]:


## set the device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
	y_train = y_train.cuda()
	y_test = y_test.cuda()
	x_coord = x_coord.cuda()




data_train = torch.utils.data.TensorDataset(y_train)
data_test = torch.utils.data.TensorDataset(y_test)


# In[11]:


z_dim = 2
print('# training with z-dim:', z_dim, file=sys.stderr)

num_layers = 2
hidden_dim = 500

activation = 'leakyrelu'
if activation == 'tanh':
	activation = nn.Tanh
elif activation == 'leakyrelu':
	activation = nn.LeakyReLU


# In[12]:


vanilla = False
_rotate = True
_translate = True

if vanilla:
	print('# using the vanilla MLP generator architecture', file=sys.stderr)
	generator_model = models.VanillaGenerator(n*m, z_dim, hidden_dim, num_layers=num_layers, activation=activation)
	inf_dim = z_dim
	rotate = False
	translate = False
else:
	print('# using the spatial generator architecture', file=sys.stderr)
	rotate = _rotate
	translate = _translate
	inf_dim = z_dim
	if rotate:
		print('# spatial-VAE with rotation inference', file=sys.stderr)
		#inf_dim += 1
	if translate:
		print('# spatial-VAE with translation inference', file=sys.stderr)
		#inf_dim += 2
	generator_model = models.SpatialGenerator(z_dim, hidden_dim, num_layers=num_layers, activation=activation,
								   resid=False, expand_coords=False)

encoder_model = models.MNIST_AttentionNetwork_groupconv_myGroupConv_2(z_dim, activation=activation)

if use_cuda:
	generator_model.cuda()
	encoder_model.cuda()


'''
path = './training_logs/2021-12-17-12-07_mnist-rotated-translated_zDim2_AttentionMaps_GroupConv/'
encoder_model = torch.load(path + '_inference_epoch021.sav').cuda()

#generator = models.VanillaGenerator(n*m, z_dim, hidden_dim, num_layers=num_layers, activation=activation)
generator_model = torch.load(path + '_generator_epoch021.sav').cuda()
'''


print(encoder_model)


num_epochs = 100

digits = int(np.log10(num_epochs)) + 1

# standard deviation on rotation prior
theta_prior = np.pi/4

rand_dist = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())

prior_rot = Normal(torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2]).unsqueeze(1).unsqueeze(2).cuda(), torch.tensor([theta_prior]*4).unsqueeze(1).unsqueeze(2).cuda())

prior_z = Normal(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())

# if 2/28 is the space between two pixels, then 0.1 is the area that covers 1.4 pixels
p_t_dist = Normal(torch.tensor([0.0]).cuda(), torch.tensor([0.3]).cuda())





N = len(mnist_train)

params = list(generator_model.parameters()) + list(encoder_model.parameters())

lr = 1e-4
optim = torch.optim.Adam(params, lr=lr)
minibatch_size = 100

train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size,
												 shuffle=True)
test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)

output = sys.stdout
print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']), file=output)





save_interval = 5

# creating a folder to save the reports and models
root = './training_logs/'
experiment_description = '_' + dataset + '_zDim'+ str(z_dim) + '_AttentionMaps_GroupConv/'
path_prefix = root + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) + experiment_description
if not os.path.exists(path_prefix):
	os.mkdir(path_prefix)
	

	
	
shutil.copy('mnist_spatialVAE_attentionMaps_GroupConv_10.py', path_prefix)
shutil.copy('spatial_vae/models.py', path_prefix)


	
train_log = []
for epoch in range(num_epochs):
	#print(optim.param_groups[0]['lr'])
	elbo_accum,gen_loss_accum,kl_loss_accum = train_epoch(train_iterator, x_coord, generator_model
														  , encoder_model, optim, rotate=rotate
														  , translate=translate, epoch=epoch
														  , num_epochs=num_epochs, N=N, use_cuda=use_cuda
														  , params=params)

	line1 = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
	output.flush()
	print(line1, file=output)
	output.flush()

	# evaluate on the test set
	elbo_accum,gen_loss_accum,kl_loss_accum = eval_model(epoch, test_iterator, x_coord, generator_model
														 , encoder_model, rotate=rotate\
														 , translate=translate,use_cuda=use_cuda)
	line2 = '\t'.join([str(epoch+1), 'test', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
	output.flush()
	print(line2, file=output)
	output.flush()
	train_log.append(line1)
	train_log.append(line2)


	## save the models
	if path_prefix is not None and (epoch+1)%save_interval == 0:
		epoch_str = str(epoch+1).zfill(digits)

		path = path_prefix + '_generator_epoch{}.sav'.format(epoch_str)
		generator_model.eval().cpu()
		torch.save(generator_model, path)

		path = path_prefix + '_inference_epoch{}.sav'.format(epoch_str)
		encoder_model.eval().cpu()
		torch.save(encoder_model, path)

		if use_cuda:
			generator_model.cuda()
			encoder_model.cuda()

with open(path_prefix + 'train_log.txt', 'w') as f:
	for l in train_log:
		f.write('%s\n' % l)





def main():
	import argparse

	parser = argparse.ArgumentParser('Train Rotation Equivariant spatial-VAE on MNIST datasets')

	parser.add_argument('--dataset', choices=['mnist', 'mnist-rotated', 'mnist-rotated-translated-notCropped', 'mnist-rotated-translated'], default='mnist-rotated-translated', help='which MNIST datset to train/validate on (default: mnist-rotated-translated)')

	parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
	parser.add_argument('--translation-inference', default='unimodal', choices=['unimodal', 'attention'], help='unimodal | attention')
	parser.add_argument('--rotation-inference', default='unimodal', help='unimodal | attention | attention+refinement' help='rotation+refinement can only be done when using the group-conv layers')
	parser.add_argument('--groupconv', type=int default=0, choices=[0, 4, 8, 16], help='0 | 4 | 8 | 16')
	parser.add_argument('--encoder-num-layers', type=int, default=2, help='number of hidden layers in original spatial-VAE inference model')				
	parser.add_argument('--encoder-kernel-number', type=int, default=128, help='number of kernels in each layer of the encoder (default: 128)')

	parser.add_argument('--image-dim', type=int, default=28, help='input image of the shape image_dim x image_dim')
	parser.add_argument('--expand-coords', action='store_true', help='using random fourier feature expansion')
	
	
	parser.add_argument('--generator-hidden-dim', type=int, default=500, help='dimension of hidden layers (default: 500)')
	parser.add_argument('--generator-num-layers', type=int, default=2, help='number of hidden layers (default: 2)')
	parser.add_argument('--generator-resid-layers', action="store_true", help='using skip connections in generator')
	parser.add_argument('--activation', choices=['tanh', 'leakyrelu'], default='leakyrelu', help='activation function (default: leakyrelu)')
	parser.add_argument('--vanilla', action='store_true', help='use the standard MLP generator architecture, decoding each pixel with an independent function. disables structured rotation and translation inference')

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
	if d >= 0:
		torch.cuda.set_device(d)
		print('# using CUDA device:', d, file=sys.stderr)
		
	   

	if use_cuda:
		y_train = y_train.cuda()
		y_test = y_test.cuda()
		x_coord = x_coord.cuda()

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
	

	if args.vanilla:
		print('# using the vanilla MLP generator architecture', file=sys.stderr)
		generator = models.VanillaGenerator(n*m, z_dim, generator_hidden_dim, num_layers=generator_num_layers,
											activation=activation, resid=generator_resid)
		inf_dim = z_dim
		rotate = False
		translate = False
	else:
		print('# using the spatial generator architecture', file=sys.stderr)
		rotate = not args.no_rotate
		translate = not args.no_translate
		inf_dim = z_dim
		if rotate:
			print('# spatial-VAE with rotation inference', file=sys.stderr)
			inf_dim += 1
		if translate:
			print('# spatial-VAE with translation inference', file=sys.stderr)
			inf_dim += 2
		generator = models.SpatialGenerator(z_dim, generator_hidden_dim, num_layers=generator_num_layers,
											activation=activation, resid=generator_resid, expand_coords=feature_expansion)
	


	# defining encoder model
	translation_inference = args.translation_inference
	rotation_inference = args.rotation_inference
	encoder_num_layers = args.encoder_num_layers
	encoder_kernel_number = args.encoder_kernel_number
	group_conv = args.groupconv

	
	if translation_inference=='unimodal' and rotation_inference=='unimodal': #original spatial-vae from Bepler et. al 2019
		encoder = models.InferenceNetwork_UnimodalTranslation_UnimodalRotation(n*n, inf_dim, encoder_kernel_number, num_layers=encoder_num_layers, activation=activation)

	elif translation_inference=='attention' and rotation_inference=='unimodal':
		encoder = models.InferenceNetwork_AttentionTranslation_UnimodalRotation_GroupConv(n, z_dim, kernels_num=encoder_kernel_number, activation=activation, groupconv=group_conv)

	elif translation_inference=='attention' and rotation_inference=='attention':
	
	
	if use_cuda:
		generator.cuda()
		encoder.cuda()

	theta_prior = args.theta_prior

	print('# using priors: theta={}, dx={}'.format(theta_prior, dx_scale), file=sys.stderr)
	print(encoder)

	N = len(mnist_train)

	params = list(generator.parameters()) + list(encoder.parameters())

	lr = args.learning_rate
	optim = torch.optim.Adam(params, lr=lr)

	minibatch_size = args.minibatch_size

	train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size,
												 shuffle=True)
	test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)
	
	output = sys.stdout
	print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']), file=output)
	
	
	#creating the log folder
	log_root = args.log_root
	if not os.path.exists(log_root):
		os.mkdir(log_root)
	experiment_description = '_'.join([str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
									  ,args.dataset, 'zDim', str(z_dim), encoder_type])
	if args.vanilla:
		experiment_description += '_vanillaGenerator'
	else:
		experiment_description += '_spatialGenerator'
	if generator_resid:
		experiment_description += '_withReslayers'
	experiment_description += ('_'+str(generator_num_layers)+'layers')
	path_prefix = os.path.join(log_root, experiment_description,'')
	
	if not os.path.exists(path_prefix):
		os.mkdir(path_prefix)   
		
	shutil.copy('train_mnist.py', path_prefix)
	shutil.copy('spatial_vae/models.py', path_prefix)
	
   
	save_interval = args.save_interval
	
	train_log = ['' for _ in range(2*num_epochs)]
	
	for epoch in range(num_epochs):

		elbo_accum,gen_loss_accum,kl_loss_accum = train_epoch(encoder_type, train_iterator, x_coord, generator, encoder,
															  optim, n, m, rotate=rotate, translate=translate,
															  dx_scale=dx_scale, theta_prior=theta_prior,
															  epoch=epoch, num_epochs=num_epochs, N=N,
															  use_cuda=use_cuda)

		line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
		train_log[2*epoch] = line
		print(line, file=output)
		output.flush()

		# evaluate on the test set
		elbo_accum,gen_loss_accum,kl_loss_accum = eval_model(encoder_type, test_iterator, x_coord, generator,
															 encoder, n, m, rotate=rotate, translate=translate,
															 dx_scale=dx_scale, theta_prior=theta_prior,
															 use_cuda=use_cuda
															)
		line = '\t'.join([str(epoch+1), 'test', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
		train_log[(2*epoch)+1] = line
		print(line, file=output)
		output.flush()

		#print('sigma is {}'.format(generator.embed_latent.sigma))
		## save the models
		if path_prefix is not None and (epoch+1)%save_interval == 0 and (epoch+1) > 70:
			epoch_str = str(epoch+1).zfill(digits)

			path = path_prefix + '_generator_epoch{}.sav'.format(epoch_str)
			generator.eval().cpu()
			torch.save(generator, path)

			path = path_prefix + '_inference_epoch{}.sav'.format(epoch_str)
			encoder.eval().cpu()
			torch.save(encoder, path)

			if use_cuda:
				generator.cuda()
				encoder.cuda()

	
	with open(path_prefix + 'train_log.txt', 'w') as f:
		f.write(experiment_description + '\n')
		if args.vanilla:
			f.write('Generator is a vanilla_generator with {} layers and {} hidden_units in each layer'.format(generator_num_layers, generator_hidden_dim))
		else:
			f.write('Generator is a spatial_generator with {} layers and {} hidden_units in each layer'.format(generator_num_layers, generator_hidden_dim))
		
		f.write('skip connections in generator: {} \n'.format(generator_resid))
		f.write('Encoder is {} \n'.format(encoder_type))
		
		f.write('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']) + '\n')
		for l in train_log:
			f.write('%s\n' % l)
			
		f.write('Encoder model: \n {}'.format(encoder))
		f.write('\nGenerator model: \n {}'.format(generator))


if __name__ == '__main__':
	main()





