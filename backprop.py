import numpy as np
from numpy.core.multiarray import ndarray
from keras.utils import to_categorical

label=np.zeros((10,1))
label[5]=1
fc2=np.random.rand(10,1)
fc1=np.random.rand(75,1)
weight_matrix=np.random.rand(10,75)
kernel1_size=5
kernel2_size=3
kernel3_size=2
num_channels1,num_channels2,num_channels3=10,5,5
s1,s2,s3=2,1,2
batch_size=32
input_channels=1

#The architecture described in CNN_np remains unchanged and this is the bakcprop code for that heterogenous architecture

def backpropogation(lr,input_channels,a1,a2,a3,kernel1,kernel2,fc1,weight_matrix,fc2,kernel1_size,kernel2_size,kernel3_size,labels,batch_size,index_matrix,num_channels1,num_channels2,num_channels3,stride1,stride2,stride3,reshaped_layer1,reshaped_layer2):
    #as we are using softmax activation and cross entropy loss function

    #for gradient of loss function w.r.t. the weight matrix b/w fc1 and fc2
    ind=np.arange(batch_size)
    labels=to_categorical(labels)
    diff=fc2-labels
    fc11=np.reshape(fc1,(fc1.shape[0],fc1.shape[1]))
    #grad_weight_matrix=np.dot((fc2-labels)[ind,:],fc11[ind,:])
    grad_weight_matrix=np.einsum('kn,km->knm',diff,fc11)
    #print(grad_weight_matrix.shape)
    grad_weight_matrix=grad_weight_matrix.mean(0)
    grad_fc2=(fc2-labels)
    grad_fc1=(np.dot(grad_fc2,weight_matrix))
    #print(grad_fc1.shape)
    grad_fc2=grad_fc2.mean(0)
    ##Gradient from max_pool
    rev_shaping=np.reshape(grad_fc1,(batch_size,num_channels3,-1,1),"C") #channels were flattened after pooling
    rev_shaping=np.pad(rev_shaping,((0,0),(0,0),(0,0),(int((kernel3_size**2)/2),int((kernel3_size**2)/2)-1)),'edge')
    #ind2=np.arange(index_matrix.shape[0])
    #print(index_matrix.shape,rev_shaping.shape)
    grad_a3=index_matrix*rev_shaping
    num_patches=np.array([int((a2.shape[2]-kernel3_size)/stride3) + 1,int((a2.shape[3]-kernel3_size)/stride3) + 1])
    grad_a3=np.reshape(grad_a3,(batch_size,num_channels3,-1,1),"C") #one vector for each channel
    #print(grad_a3.shape)
    def img2col_custom1(b,z,i,j,k=kernel3_size,s=stride3,gradient_matrix=grad_a3,patches=num_patches,last_layer=a2):
        result=np.zeros((last_layer.shape))
        row_num_in_patch=((i%(k**2))/k).astype(int,copy=True)
        col_num_in_patch=((i%(k**2))%k).astype(int,copy=True) #indexing has to start from 0
        #start_index_for_pool=0
        z=z.astype(int,copy=True)
        b = b.astype(int, copy=True)
        i=i.astype(int,copy=True)
        j=j.astype(int,copy=True)
        pixel_row_num=(0+ (s*((i/(k**2))/patches[0] -1)) + row_num_in_patch).astype(int,copy=True)
        pixel_col_num=(0+ (s*((i/(k**2))%patches[0] -1)) + col_num_in_patch).astype(int,copy=True)
        # The second term in the above two equations is just the index in the image at which the patch starts
        #print(inp_img.shape,kernel.shape)
        result[b,z,pixel_row_num,pixel_col_num]=gradient_matrix[b,z,i,j]
        ## Note: kernel does not require a third dimension as this is 2D maxpool
        return result
    grad_a2 = np.fromfunction(img2col_custom1, (batch_size,num_channels2,(num_patches[0]**2) * (kernel3_size ** 2), 1))
    num_patches1=np.array([int((a1.shape[2]-kernel2_size)/stride2) + 1,int((a1.shape[3]-kernel2_size)/stride2) + 1])
    grad_a2=np.reshape(grad_a2,(batch_size,num_channels2,a2.shape[2]*a2.shape[3]),"C")
    grad_a2=np.reshape(grad_a2,(num_channels2,batch_size*(a2.shape[2]*a2.shape[3])),"C")

    # The gradient of grad_z2 ie grad_a2 x grad_relu
    grad_z2=grad_a2
    np.place(grad_z2,grad_a2>0,1)
    np.place(grad_z2,grad_a2==0,0.5)
    np.place(grad_z2,grad_a2<0,0)
    grad_z2=grad_a2*grad_z2

    #reshaped_layer2=reshaped_layer2.transpose((0,1,3,2))
    reshaped_layer2=np.reshape(reshaped_layer2,(batch_size,reshaped_layer2.shape[2],int(reshaped_layer2.shape[3]*reshaped_layer2.shape[1])),"C")
    reshaped_layer2=np.reshape(reshaped_layer2,(int(batch_size*reshaped_layer2.shape[1]),reshaped_layer2.shape[2]),"C")

    grad_weight_vector2=np.dot(grad_z2,reshaped_layer2)
    grad_weight_vector2=np.reshape(grad_weight_vector2,(grad_weight_vector2.shape[0],num_channels1,kernel2_size,kernel2_size),"C")
    #print(grad_weight_vector2.shape)
    kernel2_mult=np.reshape(kernel2,(num_channels2,kernel2_size*kernel2_size*num_channels1),"C")
    grad_a1=np.dot(grad_z2.T,kernel2_mult).T
    grad_a1=np.reshape(grad_a1,(batch_size,num_channels1,-1,1),"C")
    def img2col_custom(b,z,i,j,k=kernel2_size,s=stride2,gradient_matrix=grad_a1,patches=num_patches1,last_layer=a1):
        result=np.zeros((last_layer.shape))
        row_num_in_patch=((i%(k**2))/k).astype(int,copy=True)
        col_num_in_patch=((i%(k**2))%k).astype(int,copy=True) #indexing has to start from 0
        #start_index_for_pool=0
        z=z.astype(int,copy=True)
        b = b.astype(int, copy=True)
        i = i.astype(int, copy=True)
        j = j.astype(int, copy=True)
        pixel_row_num=(0+ (s*((i/(k**2))/patches[0] -1)) + row_num_in_patch).astype(int,copy=True)
        pixel_col_num=(0+ (s*((i/(k**2))%patches[0] -1)) + col_num_in_patch).astype(int,copy=True)
        # The second term in the above two equations is just the index in the image at which the patch starts
        #print(inp_img.shape,kernel.shape)
        result[b,z,pixel_row_num,pixel_col_num]+=gradient_matrix[b,z,i,j]
        ## Note: kernel does not require a third dimension as this is 2D maxpool
        return result

    grad_a1_reshaped = np.fromfunction(img2col_custom, (batch_size,num_channels1,(num_patches1[0] ** 2) * (kernel2_size ** 2), 1))

    grad_a1_reshaped = np.reshape(grad_a1_reshaped, (batch_size, num_channels1, a1.shape[2] * a1.shape[3]), "C")
    grad_a1_reshaped = np.reshape(grad_a1_reshaped, (num_channels1, batch_size * (a1.shape[2] * a1.shape[3])), "C")
    #print(grad_a1_reshaped.shape)

    # The gradient of grad_z1 i.e. grad_a1 x grad_relu
    grad_z1 = grad_a1_reshaped
    np.place(grad_z1, grad_a1_reshaped > 0, 1)
    np.place(grad_z1, grad_a1_reshaped == 0, 0.5)
    np.place(grad_z1, grad_a1_reshaped < 0, 0)
    grad_z1 = grad_a1_reshaped * grad_z1

    #reshaped_layer1 = reshaped_layer1.transpose((0, 1, 3, 2))
    reshaped_layer1 = np.reshape(reshaped_layer1,
                                 (batch_size, reshaped_layer1.shape[2], reshaped_layer1.shape[3] * reshaped_layer1.shape[1]), "C")
    reshaped_layer1 = np.reshape(reshaped_layer1, (batch_size * reshaped_layer1.shape[1], reshaped_layer1.shape[2]), "C")

    grad_weight_vector1 = np.dot(grad_z1, reshaped_layer1)
    grad_weight_vector1 = np.reshape(grad_weight_vector1,(grad_weight_vector1.shape[0], input_channels, kernel1_size, kernel1_size), "C")
    #print(grad_weight_matrix.shape,grad_weight_vector2.shape,grad_weight_vector1.shape)

    #updating weights and biases when gradient is known
    weight_matrix=weight_matrix - lr*grad_weight_matrix
    weight_vector2=kernel2 - lr*grad_weight_vector2
    weight_vector1=kernel1 - lr*grad_weight_vector1

    return weight_matrix,weight_vector2,weight_vector1
