#@ type: compute
#@ parents:
#@   - func3
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma
#@   mem2:
#@     trans: mem2
#@     type: rdma
#@   mem3:
#@     trans: mem3
#@     type: rdma

from PIL import Image
import pickle
import numpy as np
import base64
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

def image_2D2DPCA(images, u, uu):
    a, b, c = images.shape
    new_images = np.ones((a, uu.shape[1], u.shape[1]))
    for i in range(a):
        Y = np.dot(uu.T, images[i,:,:])
        Y = np.dot(Y, u)
        new_images[i,:,:] = Y
    return new_images

def getNewImage(context_dict, action):
    mem_name1 = "mem1"
    trans1 = action.get_transport(mem_name1, 'rdma')
    trans1.reg(buffer_pool_lib.buffer_size)
    mem_name2 = "mem2"
    trans2 = action.get_transport(mem_name2, 'rdma')
    trans2.reg(buffer_pool_lib.buffer_size)
    mem_name3 = "mem3"
    trans3 = action.get_transport(mem_name3, 'rdma')
    trans3.reg(buffer_pool_lib.buffer_size)

    buffer_pool1 = buffer_pool_lib.buffer_pool({mem_name1:trans1}, context_dict["buffer_pool_metadata1"])
    load_image_remote = remote_array(buffer_pool1, metadata=context_dict["remote_input1"])
    image_data = load_image_remote.materialize()
    print('image_data_shape:{}'.format(image_data.shape))

    buffer_pool2 = buffer_pool_lib.buffer_pool({mem_name2:trans2}, context_dict["buffer_pool_metadata2"])
    load_u = remote_array(buffer_pool2, metadata=context_dict["remote_output2"])
    u = load_u.materialize()
    print('u_shape:{}'.format(u.shape))

    buffer_pool3 = buffer_pool_lib.buffer_pool({mem_name3:trans3}, context_dict["buffer_pool_metadata2"])
    load_uu = remote_array(buffer_pool3, metadata=context_dict["remote_output3"])
    uu = load_uu.materialize()
    print('uu_shape:{}'.format(uu.shape))

    new_image = image_2D2DPCA(image_data, u, uu)

    result_path = "new-image.jpg"
    # Save the new image
    a,b,c = new_image.shape
    new_image = new_image.reshape(b, c)
    new_im = Image.fromarray(new_image , 'L')
    new_im.save(result_path)

def main(params, action):
    # Load from previous memory
    context_dict_in_b64 = params["func3"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    getNewImage(context_dict, action)
