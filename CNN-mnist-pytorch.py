import numpy as np
import time
import matplotlib.pyplot as plt

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
    
    return training_images_vecs, training_labels, test_images_vecs, test_labels

def joint_likelihood(images, labels):
    num_classes = len(np.unique(labels))
    (num_samples, num_features) = images.shape
    conditional_probability = np.zeros((num_classes, num_features))
    count = np.zeros((num_classes,))
    prior = np.zeros((num_classes,))
    # Counting
    for i in range(num_samples):
        label_temp = labels[i]
        count[label_temp] += 1
        for j in range(num_features):
            conditional_probability[label_temp][j] += images[i, j]
            
    # Divide by the number of samples in that class
    # Avoid estimating a zero probability:
    # incorporating uniform Dirichlet prior to likelihood estimation
    # (Assuming there are some observed samples evenly spread over possible values of each feature, [0,1] in this case)
    # e.g., observed 2 samples, one for feature 1, and another one for feature 0
    # p(x_j|y_i) = (#feature_j + 1) / (#y_i + 2);    x_j = 0|1
    for i in range(num_classes):
        # maximum a posterior estimation or lapace smoothing
        conditional_probability[i, :] = (conditional_probability[i, :]+1) / (count[i]+2)  
    
    # incorporating uniform Dirichlet prior for estimating prior 
    # Assuming there are some observed samples evenly spread over possible values of each class
    # e.g. observed k samples (k is the number of classes), one sample for each category
    # p(y_i) = (#y_i+1) / (#all + k)
    for i in range(num_classes):
        prior[i] = (count[i]+1) / (num_samples+num_classes)

    return conditional_probability, prior

def predict(feature_vec_test, conditional_probability, prior):
    (num_classes, num_features) = conditional_probability.shape
    prob_max = -np.inf
    pred_class = -1
    for i in range(num_classes): 
        prob = np.log(prior[i])
        for j in range(num_features):
            feature_temp = feature_vec_test[j]
            # Avoid numerical issues (nan or inf) caused by multiplying many small float numbers
            # maximize the log function (sum) instead of original posterior probability (multiplication)
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

def generate(conditional_probability, class_i):
    (num_classes, num_features) = conditional_probability.shape
    feature_vec = np.zeros((num_features,))
    for i in range(num_features):
        feature_vec[i] = np.random.rand() < conditional_probability[class_i, i] 
    
    return feature_vec


if __name__ == '__main__':
    # Load data
    training_images_vecs, training_labels, test_images_vecs, test_labels = data_preprocess()

    # Learning
    print("Learning")
    labels = training_labels
    images = training_images_vecs
    begin = time.time()
    conditional_probability, prior = joint_likelihood(images, labels)
    print("time cost for learning:", time.time()-begin)

    # Inference
    begin = time.time()
    print("Inference")
    pred_labels = []
    for i in range(10000):
        if i%500 == 0:
            print(i)
        pred_labels.append(predict(test_images_vecs[i,:], conditional_probability, prior ))
    print("time cost for inference:", time.time()-begin)

    # Acc: 0.8433
    print("Test accuracy:", np.mean(np.array(pred_labels)==test_labels))

    # Sampling
    # Generate image for given class y_i from learned conditional probability p(x_j|y_i)
    given_class = 0
    feature_vec = generate(conditional_probability, given_class)
    img_gen = feature_vec.reshape(28,28)
    plt.imshow(img_gen, cmap='gray')
    plt.show()

    # Observe learned distribution of each feature
    feature_vec = conditional_probability[6,:]
    img_gen = feature_vec.reshape(28,28)
    plt.figure()
    plt.imshow(img_gen, cmap="jet")
    plt.show()
