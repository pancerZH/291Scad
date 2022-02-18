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

def fetch(context_dict, action):
    mem_name = "mem1"
    trans = action.get_transport(mem_name, 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)

    buffer_pool = buffer_pool_lib.buffer_pool({mem_name:trans}, context_dict["buffer_pool_metadata1"])
    load_new_image = remote_array(buffer_pool, metadata=context_dict["remote_output1"])
    new_image = load_new_image.materialize()

    print('new_images_shape:{}'.format(new_image.shape))
    return new_image

def main(params, action):
    # Load from previous memory
    context_dict_in_b64 = params["func2"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    new_image = fetch(context_dict, action)

    result_path = "new-image.jpg"

    # Save the new image
    a,b,c = new_image.shape
    new_image = new_image.reshape(b, c)
    new_im = Image.fromarray(new_image , 'L')
    new_im.save(result_path)
