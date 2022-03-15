#@ type: compute
#@ dependents:
#@   - func2
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

def processImage(im):
    mem_name = "mem1" + ".npy"

    # process image
    im_grey = im.convert('L')
    a, b = np.shape(im_grey)
    data = im_grey.getdata()
    data = np.array(data)
    data = data.reshape(1, a, b)
    print('data_shape:{}'.format(data.shape))

    with open(mem_name, 'wb') as f:
        np.save(f, data)

def main():
    # loading data
    image_path = "image.jpg"
    im = Image.open(image_path)

    context_dict = {}
    processImage(im)
    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}

if __name__ == '__main__':
    main()
