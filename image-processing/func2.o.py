#@ type: compute
#@ parents:
#@   - func1
#@ dependents:
#@   - func3
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma

import pickle
import numpy as np
import base64
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

def TwoDPCA(imgs, dim):
    a, b, c = imgs.shape
    average = np.zeros((b,c))
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    w, v = np.linalg.eigh(G_t)
    # print('v_shape:{}'.format(v.shape))
    w = w[::-1]
    v = v[::-1]

    u = v[:,:dim]
    print('u_shape:{}'.format(u.shape))
    return u

def TTwoDPCA(imgs, dim):
    u = TwoDPCA(imgs, dim)
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img, dim)
    print('uu_shape:{}'.format(uu.shape))
    return u, uu


def image_2D2DPCA(images, u, uu):
    a, b, c = images.shape
    new_images = np.ones((a, uu.shape[1], u.shape[1]))
    for i in range(a):
        Y = np.dot(uu.T, images[i,:,:])
        Y = np.dot(Y, u)
        new_images[i,:,:] = Y
    return new_images

def processImage(context_dict, action):
    mem_name = "mem1"
    trans = action.get_transport(mem_name, 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)

    buffer_pool = buffer_pool_lib.buffer_pool({mem_name:trans}, context_dict["buffer_pool_metadata1"])
    load_image_remote = remote_array(buffer_pool, metadata=context_dict["remote_input1"])
    image_data = load_image_remote.materialize()
    print('image_data_shape:{}'.format(image_data.shape))

    u, uu = TTwoDPCA(image_data, 10)
    new_image = image_2D2DPCA(image_data, u, uu)
    print('new_images_shape:{}'.format(new_image.shape))

    buffer_pool = buffer_pool_lib.buffer_pool({mem_name:trans}, context_dict["buffer_pool_metadata1"])
    remote_output = remote_array(buffer_pool, input_ndarray=new_image, transport_name=mem_name)
    # update context
    remote_input_metadata = remote_output.get_array_metadata()
    context_dict["remote_output1"] = remote_input_metadata
    context_dict["buffer_pool_metadata1"] = buffer_pool.get_buffer_metadata()

def main(params, action):
    context_dict_in_b64 = params["func1"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    processImage(context_dict, action)

    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
