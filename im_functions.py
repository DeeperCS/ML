import numpy as np
from skimage import io
from skimage import transform

import matplotlib.pyplot as plt
%matplotlib inline

########################
# Basic Image Functions
#######################

im = np.random.rand(28, 28)
# Image save
io.imsave('im.jpg', im)

# Image read
img = io.imread("im.jpg")

# Image show
plt.figure()
io.imshow(img)

# Image resize
img_new = transform.resize(img, [64, 64], preserve_range=True)
plt.figure()
io.imshow(img_new)


########################
# Grid Plot
#######################
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

# Test gray image
plt.figure()
small_im_array = np.random.rand(25, 1, 28, 28)
grid_im = grid_gray_image(small_im_array)
io.imshow(grid_im)

# Test rgb image
plt.figure()
small_im_array = np.random.rand(25, 3, 28, 28)
grid_im = grid_rgb_image(small_im_array)
io.imshow(grid_im)