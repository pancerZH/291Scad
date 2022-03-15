#@ type: compute
#@ parents:
#@   - func1
#@ dependents:
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

import pickle
import numpy as np
import base64

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
    return u

def processImage():
    mem_name1 = "mem1" + ".npy"
 
    with open(mem_name1, 'rb') as f:
        image_data = np.load(f)
        print('image_data_shape:{}'.format(image_data.shape))

        u = TwoDPCA(image_data, 10)
        print('u_shape:{}'.format(u.shape))

        mem_name2 = "mem2" + ".npy"

        with open(mem_name2, 'wb') as ff:
            np.save(ff, u)

def main():
    processImage()

if __name__ == '__main__':
    main()
