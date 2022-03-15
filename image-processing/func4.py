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

def image_2D2DPCA(images, u, uu):
    a, b, c = images.shape
    new_images = np.ones((a, uu.shape[1], u.shape[1]))
    for i in range(a):
        Y = np.dot(uu.T, images[i,:,:])
        Y = np.dot(Y, u)
        new_images[i,:,:] = Y
    return new_images

def getNewImage():
    mem_name1 = "mem1" + ".npy"
 
    with open(mem_name1, 'rb') as f:
        image_data = np.load(f)
        print('image_data_shape:{}'.format(image_data.shape))

        mem_name2 = "mem2" + ".npy"
        with open(mem_name2, 'rb') as ff:
            u = np.load(ff)
            print('u_shape:{}'.format(u.shape))

            mem_name3 = "mem3" + ".npy"
            with open(mem_name3, 'rb') as fff:
                uu = np.load(fff)
                print('uu_shape:{}'.format(uu.shape))

                new_image = image_2D2DPCA(image_data, u, uu)

                result_path = "new-image.jpg"
                # Save the new image
                print('new_image:{}'.format(new_image.shape))
                a,b,c = new_image.shape
                new_image = new_image.reshape(b, c)
                new_im = Image.fromarray(new_image , 'L')
                new_im.save(result_path)

def main():
    getNewImage()

if __name__ == '__main__':
    main()   
