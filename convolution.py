import numpy as np
import sys

from numpy.core.multiarray import ndarray

ks = 4
st = 2
chnls=3
batch_size=32
num_filters=5
'''
input_vec=tr.rand(10,10,dtype=tr.float)
weight_vec=tr.rand(3,3,dtype=tr.float)
input_vec=input_vec.numpy()
weight_vec=weight_vec.numpy()
'''
input_vec = np.random.randn(batch_size,chnls,10, 10)
weight_vec = np.random.randn(num_filters,chnls,ks, ks)
'''
print("weights: ",weight_vec,"input: ",input_vec)
'''

#print(input_vec.shape)

# An implementation of 2D convolution on 3D input.



default_output_shape = np.array(
    [((input_vec.shape[2] - weight_vec.shape[2]) / st) + 1, ((input_vec.shape[3] - weight_vec.shape[3]) / st )+ 1])
default_output_shape=np.floor(default_output_shape)

def convolve(output_spatial_shape=default_output_shape, kernel_size=ks, stride=st, input_vector=input_vec,
             weight_vector=weight_vec,num_channels=chnls,b_size=batch_size,num_ker=num_filters):
    i_shape = np.array(input_vector.shape)
    k_size: int = kernel_size  # assuming square kernel and same stride length in x and y directions
    num_patches: ndarray = np.array([((i_shape[2] - k_size) / stride) + 1, ((i_shape[3] - k_size) / stride) + 1])

    num_patches=np.floor(num_patches)
    # num_patches_col=(i_shape[1]-ks)/stride +1
    if not(np.array_equal(num_patches, output_spatial_shape)):
        print('Error: The output shape provided is not compatible with input,kernel size and stride length')
        sys.exit()

    '''each patch will have ks^2 elements so the shape of output array which is generated by the img2col custom
            method is ks^2 * no_patches, note: no_patches is equal to the output shape: ((img-k)/str +1)^2
        '''

    num_patches=num_patches.astype(int,copy=True)
    num_channels=int(num_channels)
    b_size=int(b_size)
    num_ker=int(num_ker)
    #print(num_channels,(num_patches[0]**2)*(k_size**2))
    def img2col_custom(nk,b,z,i,j,k=k_size,s=stride,inp_img=input_vector,kernel=weight_vector,patches=num_patches):
        reshape_layer=np.ones((kernel.shape))
        #print(reshape_layer.shape)
        row_num_in_patch=((i%(k**2))/k).astype(int,copy=True)
        col_num_in_patch=((i%(k**2))%k).astype(int,copy=True) #indexing has to start from 0
        #start_index_for_conv=0
        z=z.astype(int,copy=True)
        b = b.astype(int, copy=True)
        nk = nk.astype(int, copy=True)
        pixel_row_num=(0+ (s*((i/(k**2))/patches[0] -1)) + row_num_in_patch).astype(int,copy=True)
        pixel_col_num=(0+ (s*((i/(k**2))%patches[0] -1)) + col_num_in_patch).astype(int,copy=True)
        # The second term in the above two equations is just the index in the image at which the patch starts
        result=inp_img[b,z,pixel_row_num,pixel_col_num]*kernel[nk,z,row_num_in_patch,col_num_in_patch]
        reshaped_layer=inp_img[b,z,pixel_row_num,pixel_col_num]*reshape_layer[nk,z,row_num_in_patch,col_num_in_patch]
        reshaped_layer=reshaped_layer[0,:,:,:,:]
        return result,reshaped_layer
    output_col,reshaped_layer=np.fromfunction(img2col_custom, (num_ker,b_size,num_channels,(num_patches[0]**2)*(k_size**2), 1) )
    output_col=np.reshape(output_col, (num_ker,b_size,num_channels, -1 ,(k_size**2)), 'C')
    reshaped_layer = np.reshape(reshaped_layer, (b_size, num_channels, -1, (k_size ** 2)), 'C')
    output_col=(output_col.sum(4)).sum(2) # summing across columns and channels #conv operation completed at this step, reshaping left
    #print('The output shape: ',output_col.shape[2],'expected size: ',num_patches[0]**2) #should be (num_patches^2)
    #output_slice=np.zeros((num_ker,b_size,num_patches[0],num_patches[1]))
    output_slice=np.reshape(output_col, (b_size,num_ker,num_patches[0],num_patches[1]), "C")
    #print(output_slice.shape)
    #print(reshaped_layer.shape)
    return output_slice,reshaped_layer


convolve()
