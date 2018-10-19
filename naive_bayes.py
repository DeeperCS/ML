import numpy as np

def data_preprocess(train_img="train-images-idx3-ubyte", train_label="train-labels-idx1-ubyte", test_img="t10k-images-idx3-ubyte", test_label="t10k-labels-idx1-ubyte"):
    with open(train_img, 'rb') as f:
        training_images_raw = f.read()

    training_images_bytearray = bytearray(training_images_raw)
    training_images_numpy = np.array(training_images_bytearray[16:])
    training_images = training_images_numpy.reshape([-1, 28, 28])

    with open(train_label, 'rb') as f:
        training_labels_raw = f.read()

    training_labels_bytearray = bytearray(training_labels_raw)
    training_labels = np.array(training_labels_bytearray[8:])

    with open(test_img, 'rb') as f:
        test_images_raw = f.read()

    test_images_bytearray = bytearray(test_images_raw)
    test_images_numpy = np.array(test_images_bytearray[16:])
    test_images = test_images_numpy.reshape([-1, 28, 28])

    with open(test_label, 'rb') as f:
        test_labels_raw = f.read()

    test_labels_bytearray = bytearray(test_labels_raw)
    test_labels = np.array(test_labels_bytearray[8:])

    print(training_images.shape)
    print(training_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)
    
    # Reshape
    training_images_vecs = training_images.reshape(60000,-1)
    test_images_vecs = test_images.reshape(10000, -1)
    # Binarization
    training_images_vecs = 1 * (training_images_vecs>128)
    test_images_vecs = 1 * (test_images_vecs>128)
    
    return training_images_vecs, test_images_vecs, training_images_vecs, test_images_vecs

def joint_likelihood(images, labels):
    num_classes = len(np.unique(labels))
    (num_samples, num_features) = images.shape
    conditional_probability = np.zeros((num_classes, num_features))
    prior = np.zeros((num_classes,))
    for i in range(num_samples):
        label_temp = labels[i]
        prior[label_temp] += 1
        for j in range(num_features):
            conditional_probability[label_temp][j] += images[i, j]
    # Divide by the number of samples in that class
    for i in range(num_classes):
        conditional_probability[i, :] = (conditional_probability[i, :]+1) / (prior[i]+2)  # lapace smoothing
        
    return conditional_probability, prior

def predict(feature_vec_test, conditional_probability, prior):
    (num_classes, num_features) = conditional_probability.shape
    prob_max = -np.inf
    pred_class = -1
    for i in range(num_classes): 
        prob = np.log(prior[i])
        for j in range(num_features):
            feature_temp = feature_vec_test[j]
            # To avoid the result shrinking to nan or inf
            # maximize the log function (sum) instead of original posterior probility (multiplication)
            # see http://www-inst.eecs.berkeley.edu/~cs70/sp15/notes/n21.pdf
            if feature_temp==1:
                prob_feat = feature_temp * np.log(conditional_probability[i, j])
            else:
                prob_feat = (1-feature_temp) * np.log(1-conditional_probability[i, j])
            prob += prob_feat
        if prob_max<=prob:
            prob_max = prob
            pred_class = i
        
    return pred_class

import time
# Load data
training_images_vecs, training_labels, training_images_vecs, test_labels = data_preprocess()

# Learning
labels = training_labels
images = training_images_vecs
begin = time.time()
conditional_probability, prior = joint_likelihood(images, labels)
print("time cost for learning:", time.time()-begin)

# Inference
begin = time.time()

pred_labels = []
for i in range(10000):
    if i%500 == 0:
        print(i)
    pred_labels.append(predict(test_images_vecs[i,:], conditional_probability, prior ))
print("time cost for inference:", time.time()-begin)

# Acc: 0.84330000000000005
print("Test accuracy:", np.mean(np.array(pred_labels)==test_labels))
