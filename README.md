# TARGET-VAE

Source code for <a href=""> Unsupervised Object Representation Learning using Translation and Rotation Group Equivariant VAE </a>

<img src="images/model_p8_2.gif" alt="TARGET-VAE framework">


### Setup
Dependencies
<ul>
<li> Python 3 </li>
<li> Pytorch >= 1.11 </li>
<li> torchvision >= 0.12 </li>
<li> numpy >= 1.21 </li>
<li> scikit-learn >= 1.0.2 </li>
<li> astropy >= 5.0.4 </li>
</ul>


### Usage
The code in train_mnist.py, train_particles.py, train_dsprites.py, and train_galaxy.py, train TARGET-VAE on mnist (regular, MNIST(N), MNIST(U)), particle stacks of cryo-EM, dSprites, and galaxies datasets. The scripts with clustering at the start of their names, can be used to apply the trained model for clustering on a specific dataset.

For example to 
