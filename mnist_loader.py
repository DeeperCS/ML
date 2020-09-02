import numpy as np
from PIL import Image
import struct

def decode_idx3_ubyte(idx3_ubyte_file, saveFlag, status):

    '''
        idx3_ubyte_file: source file
        saveFlag: bool var (save image or not)
        status: Train or test (like 'test/') 
    '''
    with open(idx3_ubyte_file, 'rb') as f:
        buf = f.read()

    offset = 0
    magic, imageNum, rows, cols = struct.unpack_from('>IIII', buf, offset)
    offset += struct.calcsize('>IIII')
    images = np.empty((imageNum,rows, cols))
    image_size = rows * cols
    fmt = '>' + str(image_size) + 'B'

    for i in range(imageNum):

        images[i] = np.array(struct.unpack_from(fmt, buf, offset)).reshape((rows,cols))

        if saveFlag == True:
        #保存图像
            im = Image.fromarray(np.uint8(images[i]))
            im.save(status + str(i) + '.png')

        offset += struct.calcsize(fmt)

    return images

def decode_idx1_ubyte(idx1_ubyte_file):

    # idx3_ubyte_file: source file

    with open(idx1_ubyte_file, 'rb') as f:
        buf = f.read()

    offset = 0
    magic, LabelNum = struct.unpack_from('>II', buf, offset)
    offset += struct.calcsize('>II')
    Labels = np.zeros((LabelNum))

    for i in range(LabelNum):

        Labels[i] = np.array(struct.unpack_from('>B', buf, offset))
        offset += struct.calcsize('>B')

    return Labels
    
    

train_images = decode_idx1_ubyte("../data/MNIST/raw/train-images-idx3-ubyte")
test_images = decode_idx1_ubyte("../data/MNIST/raw/t10k-images-idx3-ubyte")

train_lables = decode_idx1_ubyte("../data/MNIST/raw/train-labels-idx1-ubyte")
test_lables = decode_idx1_ubyte("../data/MNIST/raw/t10k-labels-idx1-ubyte")
