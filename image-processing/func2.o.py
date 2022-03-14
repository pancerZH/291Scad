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

def processImage(context_dict, action):
    mem_name1 = "mem1"
    trans1 = action.get_transport(mem_name1, 'rdma')
    trans1.reg(buffer_pool_lib.buffer_size)

    buffer_pool1 = buffer_pool_lib.buffer_pool({mem_name1:trans1}, context_dict["buffer_pool_metadata1"])
    load_image_remote = remote_array(buffer_pool1, metadata=context_dict["remote_input1"])
    image_data = load_image_remote.materialize()
    print('image_data_shape:{}'.format(image_data.shape))

    u = TwoDPCA(image_data, 10)
    print('u_shape:{}'.format(u.shape))

    mem_name2 = "mem2"
    trans2 = action.get_transport(mem_name2, 'rdma')
    trans2.reg(buffer_pool_lib.buffer_size)

    buffer_pool2 = buffer_pool_lib.buffer_pool({mem_name2:trans2})
    remote_output = remote_array(buffer_pool2, input_ndarray=u, transport_name=mem_name2)
    # update context
    remote_input_metadata = remote_output.get_array_metadata()
    context_dict["remote_output2"] = remote_input_metadata
    context_dict["buffer_pool_metadata2"] = buffer_pool2.get_buffer_metadata()

def main(params, action):
    context_dict_in_b64 = params["func1"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    processImage(context_dict, action)

    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
