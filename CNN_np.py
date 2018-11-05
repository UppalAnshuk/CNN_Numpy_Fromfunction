from convolution import convolve
from pooling import max_pool
from backprop import backpropogation
from relu_activation import relu
import numpy as np
import torch as tr
import torchvision
import torch.nn as nn
import Mnist_load

chnls_input=1
batch_size=32
input_vec=np.random.randn(batch_size,chnls_input,28,28)
kernel_size1,kernel_size2,kernel_size3=5,3,2
num_filters1,num_filters2=10,5
stride1,stride2,stride3=2,1,2
weight_vec1=np.random.randn(num_filters1,chnls_input,kernel_size1,kernel_size1)
weight_vec2=np.random.randn(num_filters2,num_filters1,kernel_size2,kernel_size2)
fc2_dim=np.array([32,10,1])
train_loader,test_loader=Mnist_load.load()
weight_matrix = np.random.randn(fc2_dim[1], 125)

#data_iter = iter(train_loader)
#images, labels = data_iter.next()
#print(labels.shape,images.shape)
#images=images.numpy()
##The network
#Feature extractor
counter=10
while(counter!=0):
    for images, labels in train_loader:
        images=images.numpy()
        a1, reshaped_layer1 = convolve((12, 12), kernel_size1, stride1, images / 255, weight_vec1, chnls_input,
                                       batch_size,
                                       num_filters1)
        a1 = relu(a1)
        a2, reshaped_layer2 = convolve((10, 10), kernel_size2, stride2, a1, weight_vec2, num_filters1, batch_size,
                                       num_filters2)
        a2 = relu(a2)
        a3, i_m = max_pool((5, 5), kernel_size3, stride3, a2, num_filters2, batch_size)
        # print(reshaped_layer2.shape)
        # print("the  shape of input and layers: ", input_vec.shape, a1.shape, a2.shape, a3.shape)

        # the classifier or the model
        fc1 = np.reshape(a3, (batch_size, -1, 1), "C")
        # print(fc1.shape)

        i = np.arange(batch_size)
        fc2 = np.dot(weight_matrix, fc1[i, :, :])  # seperate multiplication for the images in the batch
        fc2 = fc2.transpose((1, 0, 2))
        # print(fc2.shape)
        fc2 = np.reshape(fc2, (batch_size, fc2_dim[1]), "C")
        ## Calculating loss Pytorch module nn used below
        fc2 = tr.from_numpy(fc2)
        # g_truth=tr.from_numpy(tral[0,:])
        sft_activation = nn.Softmax(dim=1)
        fc21 = sft_activation(fc2)

        # fc2=fc2.numpy()
        # print(fc2)
        losf = nn.CrossEntropyLoss()

        Loss = losf(fc2, labels)
        print("The loss : ", Loss)

        # propogating back after the loss is known

        weight_matrix, weight_vec2, weight_vec1 = backpropogation(0.015, chnls_input, a1, a2, a3, weight_vec1,
                                                                  weight_vec2,
                                                                  fc1, weight_matrix, fc2.numpy(), kernel_size1,
                                                                  kernel_size2, kernel_size3, labels.numpy(),
                                                                  batch_size,
                                                                  i_m, a1.shape[1], a2.shape[1], a3.shape[1], stride1,
                                                                  stride2, stride3, reshaped_layer1, reshaped_layer2)
        # print(weight_matrix.shape, weight_vec2.shape, weight_vec1.shape)

    counter = counter - 1