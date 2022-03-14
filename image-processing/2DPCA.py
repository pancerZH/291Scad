import numpy as np
from PIL import Image
from memory_profiler import profile
'''
imgs 是三维的图像矩阵，第一维是图像的个数
'''
@profile
def TwoDPCA(imgs, dim):
    a,b,c = imgs.shape
    average = np.zeros((b,c))
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    w,v = np.linalg.eigh(G_t)
    # print('v_shape:{}'.format(v.shape))
    w = w[::-1]
    v = v[::-1]

    u = v[:,:dim]
    return u


@profile
def TTwoDPCA(imgs, u, dim):
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img, dim)
    return uu


@profile
def image_2D2DPCA(image, u, uu):
    a, b, c = image.shape
    new_image = np.ones((a, uu.shape[1], u.shape[1]))
    for i in range(a):
        Y = np.dot(uu.T, image[i,:,:])
        Y = np.dot(Y, u)
        new_image[i,:,:] = Y
    return new_image


@profile
def process_image(im):
    im_grey = im.convert('L')
    # im_grey.save('a.png')
    a, b = np.shape(im_grey)
    data = im_grey.getdata()
    data = np.array(data)
    data2 = data.reshape(1, a, b)
    return data2


@profile
def save_new_image(new_image):
    result_path = "new-image.jpg"

    # Save the new image
    a,b,c = new_image.shape
    new_image = new_image.reshape(b, c)
    new_im = Image.fromarray(new_image , 'L')
    new_im.save(result_path)


if __name__ == '__main__':
    im = Image.open('./image.jpg')

    image_data = process_image(im)
    print('data2_shape:{}'.format(image_data.shape))

    u = TwoDPCA(image_data, 10)
    print('data2_2DPCA_u:{}'.format(u.shape))

    uu = TTwoDPCA(image_data, u, 10)
    print('data2_2D2DPCA_uu:{}'.format(uu.shape))

    new_image = image_2D2DPCA(image_data, u, uu)
    print('new_images:{}'.format(new_image.shape))
    save_new_image(new_image)