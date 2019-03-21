import numpy as np

# batch generator
def batch_generator(X, y, batch_size): 
    batch_count = 0
    data_size = len(X)
    # if the end of this batch exceeds the size of dataset
    while batch_count*batch_size < data_size:
        begin = batch_count*batch_size
        end = begin + batch_size
        batch_count += 1
        yield (X[begin:end, ...], y[begin:end, ...])
        

        
images = np.random.rand(1000, 1, 28, 28)
labels = np.random.rand(1000,)
epoch_num = 8
for epoch in range(epoch_num):
    print("Epoch:",epoch)
    batch_gen = batch_generator(images, labels, batch_size=256)
    for idx, (X, y) in enumerate(batch_gen):
        print("idx:", idx, "  X.shape:", X.shape, "  y.shape:", y.shape)