#@ type: compute
#@ parents:
#@   - func2
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma

from PIL import Image
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

def TTwoDPCA(imgs, u, dim):
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img, dim)
    print('uu_shape:{}'.format(uu.shape))
    return uu

def processImage(context_dict, action):
    mem_name1 = "mem1"
    trans1 = action.get_transport(mem_name1, 'rdma')
    trans1.reg(buffer_pool_lib.buffer_size)
    mem_name2 = "mem2"
    trans2 = action.get_transport(mem_name2, 'rdma')
    trans2.reg(buffer_pool_lib.buffer_size)

    buffer_pool1 = buffer_pool_lib.buffer_pool({mem_name1:trans1}, context_dict["buffer_pool_metadata1"])
    load_image_remote = remote_array(buffer_pool1, metadata=context_dict["remote_input1"])
    image_data = load_image_remote.materialize()
    print('image_data_shape:{}'.format(image_data.shape))

    buffer_pool2 = buffer_pool_lib.buffer_pool({mem_name2:trans2}, context_dict["buffer_pool_metadata2"])
    load_u = remote_array(buffer_pool2, metadata=context_dict["remote_input2"])
    u = load_u.materialize()
    print('u_shape:{}'.format(u.shape))

    uu = TTwoDPCA(image_data, u, 10)
    print('uu_shape:{}'.format(uu.shape))

    mem_name3 = "mem3"
    trans3 = action.get_transport(mem_name3, 'rdma')
    trans3.reg(buffer_pool_lib.buffer_size)

    buffer_pool3 = buffer_pool_lib.buffer_pool({mem_name3:trans3}, context_dict["buffer_pool_metadata3"])
    remote_output = remote_array(buffer_pool3, input_ndarray=uu, transport_name=mem_name3)
    # update context
    remote_input_metadata = remote_output.get_array_metadata()
    context_dict["remote_output3"] = remote_input_metadata
    context_dict["buffer_pool_metadata3"] = buffer_pool3.get_buffer_metadata()

def main(params, action):
    # Load from previous memory
    context_dict_in_b64 = params["func2"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    processImage(context_dict, action)

    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
