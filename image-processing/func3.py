#@ type: compute
#@ parents:
#@   - func2
#@ dependents:
#@   - func4
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

def TTwoDPCA(imgs, u, dim):
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img, dim)
    return uu

def processImage():
    mem_name1 = "mem1" + ".npy"
 
    with open(mem_name1, 'rb') as f:
        image_data = np.load(f)
        print('image_data_shape:{}'.format(image_data.shape))

        mem_name2 = "mem2" + ".npy"
        with open(mem_name2, 'rb') as ff:
            u = np.load(ff)
            print('u_shape:{}'.format(u.shape))

            uu = TTwoDPCA(image_data, u, 10)
            print('uu_shape:{}'.format(uu.shape))

            mem_name3 = "mem3" + ".npy"
            with open(mem_name3, 'wb') as fff:
                np.save(fff, uu)

def main():
    processImage()

if __name__ == '__main__':
    main()
