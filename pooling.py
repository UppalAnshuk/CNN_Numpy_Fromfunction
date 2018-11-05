import numpy as np
import sys

ks = 2
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
weight_vec = np.ones((ks, ks), int)
'''
print("weights: ",weight_vec,"input: ",input_vec)
'''

#type(weight_vec), type(input_vec)

# This a Maxpool 2d implementation i.e. it does not pool across depth


"""
def convolve(i,j,kernel_size=ks,stride=st,input_vector=input_vec,weight_vector=weight_vec):
  if ((input_vector.shape[0]-weight_vector.shape[0])/stride)-int((input_vector.shape[0]-weight_vector.shape[0])/stride)>0:
    input_vector=np.pad(input_vector,((0,1),(0,1)),'edge')
  output_slice_size=np.array([np.ceil((input_vector.shape[0]-weight_vector.shape[0])/stride)+1,np.ceil((input_vector.shape[1]-weight_vector.shape[1])/stride)+1],np.int32)
 #output_slice_size.astype(int,copy=False)
  print(output_slice_size,np.shape(input_vector))
  output=np.zeros([output_slice_size[0],output_slice_size[1]])
  next_layer=np.zeros_like(output)
  k=i.astype(int,copy=True)
  l=j.astype(int,copy=True)
  #print(k," ",l)
  n=(i*stride).astype(int,copy=True)
  m=(j*stride).astype(int,copy=True)
  n1=n+2
  m1=m+2
  #print(l.dtype)
  #print(k.dtype)
 #print(n1,' ',m1)
  print(input_vector[n:n1,m:m1])

  if(np.array_equal(input_vector[n,m],input_vector[:8,:8])):
    print('yes')

 #output[k,l]=(weight_vector*input_vector[n:n+weight_vector.shape[0],m:m+weight_vector.shape[1]]).sum()    


  print('output shape :',np.shape(output),'output :',output)
  return output

np.fromfunction(convolve,(8,8))


"""
default_output_shape = np.array(
    [((input_vec.shape[2] - weight_vec.shape[0]) / st) + 1, ((input_vec.shape[3] - weight_vec.shape[1]) / st) + 1])

## The output shape is always a floor of the above computation
default_output_shape = np.floor(default_output_shape)


def max_pool(output_spatial_shape=default_output_shape, kernel_size=ks, stride=st, input_vector=input_vec,num_channels=chnls,b_size=batch_size):


    i_shape = np.array(input_vector.shape)
    k_size: int = kernel_size  # assuming square kernel and same stride length in x and y directions
    weight_vector = np.ones((k_size, k_size), int)
    num_patches: ndarray = np.array([((i_shape[2] - k_size) / stride) + 1, ((i_shape[3] - k_size) / stride) + 1])

    num_patches = np.floor(num_patches)
    # num_patches_col=(i_shape[1]-ks)/stride +1
    if not (np.array_equal(num_patches, output_spatial_shape)):
        print('Error: The output shape provided is not compatible with input,kernel size and stride length')
        sys.exit()

    '''each patch will have ks^2 elements so the shape of output array which is generated by the img2col custom
            method is ks^2 * no_patches, note: no_patches is equal to the output shape: ((img-k)/str +1)^2
        '''
    num_patches = num_patches.astype(int, copy=True)
    #print(num_patches)
    num_channels=int(num_channels)
    b_size=int(b_size)
    def img2col_custom(b,z,i,j,k=k_size,s=stride,inp_img=input_vector,kernel=weight_vector,patches=num_patches):

        row_num_in_patch=((i%(k**2))/k).astype(int,copy=True)
        col_num_in_patch=((i%(k**2))%k).astype(int,copy=True) #indexing has to start from 0
        #start_index_for_pool=0
        z=z.astype(int,copy=True)
        b = b.astype(int, copy=True)
        pixel_row_num=(0+ (s*((i/(k**2))/patches[0] -1)) + row_num_in_patch).astype(int,copy=True)
        pixel_col_num=(0+ (s*((i/(k**2))%patches[0] -1)) + col_num_in_patch).astype(int,copy=True)
        # The second term in the above two equations is just the index in the image at which the patch starts
        #print(inp_img.shape,kernel.shape)
        result=inp_img[b,z,pixel_row_num,pixel_col_num]*kernel[row_num_in_patch,col_num_in_patch]
        ## Note: kernel does not require a third dimension as this is 2D maxpool
        return result

    output_col = np.fromfunction(img2col_custom, (b_size,num_channels,(num_patches[0] ** 2) * (k_size ** 2), 1))
    output_col = np.reshape(output_col, (b_size,num_channels, -1, (k_size ** 2)), 'C')

    index_matrix=np.zeros_like(output_col)
    index_matrix[np.isin(output_col,output_col.max(3))==True]=1

    output_col = output_col.max(3)  # max across columns #pooling operation completed at this step, reshaping left

    #print('The output shape: ', output_col.shape[2], 'expected size: ', num_patches[0] ** 2)  # should be (num_patches^2)
    #output_slice = np.zeros((b_size,num_channels,num_patches[0],num_patches[1]))
    output_slice = np.reshape(output_col, (b_size,num_channels,num_patches[0],num_patches[1]), "C")
    #print(output_slice.shape)
    #print(index_matrix.shape)
    return output_slice,index_matrix #use np.nonzero for finding elements


#max_pool()