'''
    Image denoiser tools

'''

import skimage.color
import skimage.io
import numpy as np


def addNoise(image, sigma):

    if image.ndim == 2:

        n = np.random.normal(0, sigma, image.shape)
        img_n = np.float32(image) + n
        img_n = np.float32(img_n)
        img_n[img_n>255.0] = 255.0
        img_n[img_n<0] = 0
        return img_n

    elif image.ndim == 3:

        ycbcr = skimage.color.rgb2ycbcr(image)
        img = ycbcr[:,:,0]
        n = np.random.normal(0, sigma, img.shape)
        img_n = img + n
        img_n[img_n>255.0] = 255.0
        img_n[img_n<0] = 0
        ycbcr[:,:,0] = img_n
        img_c = skimage.color.ycbcr2rgb(ycbcr)
        img_c[img_c>1.0] = 1.0
        img_c[img_c<0] = 0
        img_c = np.float32(img_c)
        return img_c


def mse_psnr(image_ref, image):

    if image_ref.ndim == 2 and image.ndim == 2:
        scale = 255
        scale_ref = 255

        if np.max(image)>1:
            scale = 1
        if np.max(image_ref)>1:
            scale_ref = 1

        mse = np.mean(np.square(image*scale - np.float32(image_ref)*scale_ref))
    elif image_ref.ndim == 3 and image.ndim == 3:
        g_ref = skimage.color.rgb2gray(image_ref)
        g_o = skimage.color.rgb2gray(image)
        mse = 255*255*np.mean(np.square(g_ref - g_o))

    psnr = 10*np.log10(255*255/mse)

    return mse, psnr