import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

    
def downBlock(input_img,input_channel,inner_channel,output_channel):

    tower_1 = Conv2D(inner_channel,kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu',dilation_rate=(1,1))(input_img)

    tower_2 = Conv2D(inner_channel,kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu',dilation_rate=(3,3))(input_img)

    tower_3 = Conv2D(inner_channel,kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu',dilation_rate=(5,5))(input_img)

    output1 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    conv4 =  Conv2D(output_channel,kernel_size=(3,3),strides=(2,2),padding='same',activation='relu')(output1)

    conv5 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1))(conv4)

    out = BatchNormalization()(conv5)

    out = Activation(activation='relu')(out)

    return out

def denseBlock(input_img,input_channel,growth_rate,output_channel):

    conv1 = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1) , padding='same' )(input_img)
    b1 = BatchNormalization()(conv1)
    r1 = Activation(activation='relu')(b1)

    concat1 = keras.layers.concatenate([input_img,r1],axis=1)

    conv2 = Conv2D(growth_rate,kernel_size(3,3),strides=(1,1), padding='same')(concat1)
    b2 = BatchNormalization()(conv2)
    r2 = Activation(activation='relu')(b2)

    concat2 = keras.layers.concatenate([concat1,r2],axis=1)

    conv3 = Conv2D(output_channel,kernel_size(3,3),strides=(1,1), padding='same')(concat2)
    b3 = BatchNormalization()(conv3)
    r3 = Activation(activation='relu')(b3)

    return r3


def upBlock(input_img,input_channel,inner_channel,output_channel,kernel_size):

    conv1 = Conv2D(inner_channel,kernel_size=(3,3),strides=(1,1),padding='same')(input_img)
    r1 = Activation(activation='relu')(conv1)

    conv2T = Conv2DTranspose(output_channel,kernel_size=kernel_size,strides=(2,2),padding='same')(r1)
    r2 = Activation(activation='relu')(conv2T)

    conv3 = Conv2D(output_channel,kernel_size(3,3),strides=(1,1),padding='same')(r2)
    b = BatchNormalization()(conv3)
    r3 = Activation(activation='relu')(b)

    return r3

def transitionBlock(input_img,input_channel,output_channel):

    conv1 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(input_img)

    pool1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(input_img)
    conv2 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(pool1)

    conv31 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(input_img)
    conv32 = Conv2D(output_channel,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(conv31)

    conv41 = Conv2D(output_channel,kernel_size=(1,1),strides=(1,1),activation='relu')(input_img)
    conv42 = Conv2D(output_channel,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')(conv41)

    concat = keras.layers.concatenate([conv1,conv2,conv32,conv42],axis=1)

    convf = Conv2D(output_channel,kernel_size=(3,3),strides=(1,1),padding='same')(concat)

    b = BatchNormalization()(convf)
    r = Activation(activation='relu')(b)

    return r

def baseUNet(input_size=(512,512,1)):
    

    