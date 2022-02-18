#@ type: compute
#@ dependents:
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

def processDataSlice(data, context_dict, action):
    mem_name = "mem1"
    trans = action.get_transport(mem_name, 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)
    buffer_pool = buffer_pool_lib.buffer_pool({mem_name:trans})

    print('data_shape:{}'.format(data.shape))

    remote_input = remote_array(buffer_pool, input_ndarray=data, transport_name=mem_name)

    # update context
    remote_input_metadata = remote_input.get_array_metadata()

    context_dict["remote_input1" ] = remote_input_metadata
    context_dict["buffer_pool_metadata1"] = buffer_pool.get_buffer_metadata()

def main(context_dict, action):
    # loading data
    image_path = "image.jpg"
    im = Image.open(image_path)
    im_grey = im.convert('L')
    a, b = np.shape(im_grey)
    data = im_grey.getdata()
    data = np.array(data)
    data2 = data.reshape(1, a, b)
    print('data2_shape:{}'.format(data2.shape))

    context_dict = {}
    processDataSlice(data2, context_dict, action)
    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
