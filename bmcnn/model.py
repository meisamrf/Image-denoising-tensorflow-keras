'''
Block matching CNN based Image denoiser inspired by https://arxiv.org/abs/1704.03264

Differences:
    1. We added block matching
    2. Lighter network
    3. Single model

'''


import keras.models as KM
import keras.layers as KL
import numpy as np
import skimage.color
import blockmatch

      
def denoiser_graph(input_image):

    x = KL.Conv2D(64, kernel_size=(3, 3), padding='same', dilation_rate = (1,1))(input_image)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(24, kernel_size=(3, 3), padding='same', dilation_rate = (2,2))(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(24, kernel_size=(3, 3), padding='same', dilation_rate = (3,3))(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(24, kernel_size=(3, 3), padding='same', dilation_rate = (4,4))(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(24, kernel_size=(3, 3), padding='same', dilation_rate = (3,3))(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(24, kernel_size=(3, 3), padding='same', dilation_rate = (2,2))(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(1, kernel_size=(3, 3), activation= None, padding='same', dilation_rate = (1,1))(x)
    return x


class ImageDenoiser:
    def __init__(self, model_weights=None):
        '''
        Constructor of ImageDenoiser.
        :param model_weights (str): location of saved model weights
        :return:
        '''
        # define model
        self.model = None
        input_shape = (None, None, 5)
        input_image = KL.Input(input_shape, dtype="float32")
        output_image = denoiser_graph(input_image)
        self.model = KM.Model(inputs=input_image, outputs=output_image)
        self.base_sigma = 15.0
        # load weights
        if model_weights == None:
            model_weights = "./model_ver1.0.h5"
    
        self.model.load_weights(model_weights)

    def run(self, image, sigma):
        '''
        Run the denoiser on the image.
        :param image (uint8, float32 2D or 3D numpy array) : can be gray or RGB (values between 0 and 255)
        :param sigma (float number) : standard deviation of noise
        '''
        
        scale_adjust = min(max(sigma/self.base_sigma, 0.2), 5)
        # check image format
        ycbcr = None
        if type(image) != np.ndarray:
            print('error')
            return None

        if image.ndim == 3:
            ycbcr = skimage.color.rgb2ycbcr(image)
            gray_image = np.float32(ycbcr[:,:,0])
        else:
            gray_image = image

        if gray_image.dtype == np.uint8:
            gray_image = np.float32(gray_image)/255.0
        elif gray_image.dtype == np.float32:
            gray_image = gray_image/255.0
        else:    
            print('error')
            return None

        gray_image /= scale_adjust

        x_hat = np.zeros((1, gray_image.shape[0], gray_image.shape[1], 5), np.float32)
        gray_image_3d = np.zeros((gray_image.shape[0], gray_image.shape[1], 5), np.float32)
        blockmatch.run(gray_image, gray_image_3d)
        x_hat[0] = gray_image_3d
        n_hat = self.model.predict(x_hat)
        img_out = gray_image - np.reshape(n_hat[0], (gray_image.shape[0], gray_image.shape[1]))

        img_out *= scale_adjust
        img_out[img_out<0.0] = 0.0
        img_out[img_out>1.0] = 1.0
        
        if ycbcr is not None:
            ycbcr[:,:,0] = img_out*255
            img_out = skimage.color.ycbcr2rgb(ycbcr)
            img_out[img_out<0.0] = 0.0
            img_out[img_out>1.0] = 1.0
        return img_out
