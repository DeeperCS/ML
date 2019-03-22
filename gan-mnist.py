####################################
# GAN on MNIST
###################################

from skimage import io
import matplotlib.pyplot as plt
# %matplotlib inline

import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# batch generator
def batch_generator(X, y, batch_size): 
    batch_count = 0
    data_size = len(X)
    
    # if the end of this batch exceeds the size of dataset
    while batch_count*batch_size < data_size:
        # print(batch_count*batch_size)
        # import pdb; pdb.set_trace()
        begin = batch_count*batch_size
        end = begin + batch_size
        
        batch_count += 1
        
        yield (X[begin:end, ...], y[begin:end, ...])

def grid_gray_image(im_npy):
    # input:
    # size of im_npy: (batch_size, channels, height, width) 
    # output:
    # size of image: im_height, im_width
    
    # grid padding
    padding = 2
    # grid size
    grid_counts_x = 5
    grid_counts_y = 5

    [im_count, channel, height, width] = im_npy.shape
    im_height = (height+padding) * grid_counts_y
    im_width = (width+padding) * grid_counts_y
    image = np.zeros((im_height, im_width))
    
    for i in range(grid_counts_x):
        for j in range(grid_counts_y):
            image[(height+padding)*i:(height+padding)*i+height, (width+padding)*j:(width+padding)*j+width] =\
            np.squeeze(im_npy[i*grid_counts_x+j, ...])
            
    return image

def grid_rgb_image(im_npy):
    # input:
    # size of im_npy: (batch_size, channels, height, width) 
    # output:
    # size of image: im_height, im_width
    
    # grid padding
    padding = 2
    # grid size
    grid_counts_x = 5
    grid_counts_y = 5

    [im_count, channel, height, width] = im_npy.shape
    im_height = (height+padding) * grid_counts_y
    im_width = (width+padding) * grid_counts_y
    image = np.zeros((channel, im_height, im_width))
    
    for i in range(grid_counts_x):
        for j in range(grid_counts_y):
            image[:, (height+padding)*i:(height+padding)*i+height, (width+padding)*j:(width+padding)*j+width] =\
            np.squeeze(im_npy[i*grid_counts_x+j, ...])
            
    return image.transpose([1,2,0])

class Parameters():
    pass

opt = Parameters()

opt.img_size = 28
opt.channels = 1

opt.latent_dim = 100
opt.batch_size = 64
opt.n_epochs = 200

opt.lr = 0.0002
opt.b1 = 0.5
opt.b2 = 0.999

opt.sample_interval = 200

# GPU id
device_id = 0

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



####################################
# Generator and Discriminator
###################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.init_size = opt.img_size // (2**2)         # 28/4=7
        self.linear = nn.Linear(opt.latent_dim, 128*self.init_size**2) # 1 * 128 * 7 * 7
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, opt.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(opt.channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.output = nn.Sequential(
            nn.Linear(32*(opt.img_size//(2**2))**2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        # import pdb; pdb.set_trace()
        out = self.output(out)
        return out
    
    
generator = Generator()
discriminator = Discriminator()
adversarial_loss = torch.nn.BCELoss()

if cuda:
    torch.cuda.set_device(device_id)
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
print(generator(Variable(torch.randn(10,100)).cuda()).size())
print(discriminator(Variable(torch.randn(10,1,28,28)).cuda()).size())

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


####################################
# Load Data
###################################
data = datasets.MNIST('../../data/mnist', train=True, download=True)
images = data.data.numpy().reshape(60000, 1, 28, 28)
label = np.zeros((60000))
print(images.shape)


####################################
# Training and Save Images
###################################
# Configure data loader
save_dir = './mnist_gen1/'
os.makedirs(save_dir, exist_ok=True)

num_epoch = 50
batch_size = 128
batch_done = 0
g_loss_arr = []
d_loss_arr = []
for epoch in range(num_epoch):
    batch_gen = batch_generator(images, label, batch_size=batch_size)
    for idx, (X, y) in enumerate(batch_gen):
        data_len = X.shape[0]

        real_label = Variable(Tensor(data_len, 1).fill_(1.0), requires_grad=False)
        fake_label = Variable(Tensor(data_len, 1).fill_(0.0), requires_grad=False)

        real_images = Variable(Tensor(X))
        noise = Variable(Tensor(np.random.normal(0, 1, (data_len, opt.latent_dim))))
        fake_images = generator(noise)

        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(fake_images), real_label)
        g_loss.backward()
        optimizer_G.step()    

        optimizer_D.zero_grad()
        d_loss = adversarial_loss(discriminator(fake_images.detach()), fake_label) + \
                    adversarial_loss(discriminator(real_images), real_label)
        d_loss.backward()
        optimizer_D.step()

        if batch_done%20==0:
            print("epoch:{}, batch_done:{}, g_loss:{}, d_loss:{}".format(epoch, batch_done, g_loss.item(), d_loss.item()))
        g_loss_arr.append(g_loss.item())
        d_loss_arr.append(d_loss.item())
        if batch_done%opt.sample_interval == 0:
            grid_im = grid_gray_image(fake_images.data.cpu().numpy())
            io.imsave(save_dir+'grid_im-{}.jpg'.format(batch_done), grid_im)
            
        batch_done += 1