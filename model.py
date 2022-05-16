# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 02:21:06 2022

@author: Muaz
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras
from matplotlib import pyplot as plt
import glob
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras.metrics import MeanIoU

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import keras.backend as kb



def fmbconv(x, filter_size, size, dropout_rate, batch_norm=False):
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size),padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=4)(conv)

    conv = squeeze_excite_block(conv)


    conv = layers.Conv3D(size, kernel_size=(1, 1, 1), padding='same')(conv)

    if batch_norm is True:
        conv = layers.BatchNormalization(axis=4)(conv)

    if dropout_rate > 0:
        print(dropout_rate, 'dropout_rate')
        conv = layers.Dropout(dropout_rate)(conv)
    
    shortcut = layers.Conv3D(size, kernel_size=(1, 1, 1), padding='same')(x)

    res_path = layers.add([conv, shortcut])
    res_path = layers.Activation('relu')(res_path)
    return res_path


def res_conv_block(x, filter_size, size, dropout_rate, batch_norm=False):
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=4)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=4)(conv)


    shortcut = layers.Conv3D(size, kernel_size=(1, 1, 1), padding='same')(x)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)

    if dropout_rate > 0:
        print(dropout_rate, 'dropout_rate')
        conv = layers.Dropout(dropout_rate)(conv)
    return res_path



# Credits : https://github.com/laugh12321/3D-Attention-Keras

class channel_attention(tf.keras.layers.Layer):
    """ 
    channel attention module 
    
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, ratio=8, **kwargs):
        self.ratio = ratio
        super(channel_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(channel_attention, self).get_config().copy()
        config.update({
            'ratio': self.ratio
        })
        return config

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channel // self.ratio,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = tf.keras.layers.GlobalAveragePooling3D()(inputs)    
        avg_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = tf.keras.layers.GlobalMaxPooling3D()(inputs)
        max_pool = tf.keras.layers.Reshape((1, 1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = tf.keras.layers.Add()([avg_pool, max_pool])
        feature = tf.keras.layers.Activation('sigmoid')(feature)

        return tf.keras.layers.multiply([inputs, feature])


class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=7, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(spatial_attention, self).get_config().copy()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

    def build(self, input_shape):
        self.conv3d = tf.keras.layers.Conv3D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv3d(concat)	
            
        return tf.keras.layers.multiply([inputs, feature])


def cbam_block(feature, ratio=8, kernel_size=7):
    """
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    feature = channel_attention(ratio=ratio)(feature)
    feature = spatial_attention(kernel_size=kernel_size)(feature)

    return feature


# Credits : https://github.com/bnsreenu/python_for_microscopists/blob/master/224_225_226_models.py#L136

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    
    
    theta_x = layers.Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)
    
    phi_g = layers.Conv3D(inter_shape, (1, 1, 1), padding='same')(gating)
    
    upsample_g = layers.Conv3D(inter_shape, (3, 3, 3),
                                 padding='same')(phi_g)

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    
    psi = layers.Conv3D(1, (1, 1, 1), padding='same')(act_xg)
    
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    
    upsample_psi = layers.UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)
    
    upsample_psi = repeat_elem(upsample_psi, shape_x[4])
    #print(K.int_shape(upsample_psi))
    y = layers.multiply([upsample_psi, x])

    result = layers.Conv3D(shape_x[4], (1, 1, 1), padding='same')(y)
    result = layers.BatchNormalization()(result)

    result = layers.Conv3D(shape_x[4], (3, 3, 3), padding='same')(result)

    result = cbam_block(result)
    return result



def gating_signal(input, out_size, batch_norm=True):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv3D(out_size, (1, 1, 1), activation='relu', padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return x


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 4
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    #se = GlobalAveragePooling3D()(init)
    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    print(se.shape, 'se shape')
    se = Dense(filters // ratio, activation='relu',  use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)

    x = multiply([init, se])
    # x = Dropout(0.5)(x)
    return x


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
                          arguments={'repnum': rep})(tensor)

def _model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes, dropout_rate=0.3, batch_norm=True):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs
    filter_size = 3

    FILTER_NUM = 16
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2
    

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    conv_128 = fmbconv(inputs, FILTER_SIZE, FILTER_NUM, 0.1, batch_norm)
    pool_64 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_128)

    conv_64 = fmbconv(pool_64, FILTER_SIZE, 2*FILTER_NUM, 0.2, batch_norm)
    pool_32 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_64)

    conv_32 = fmbconv(pool_32, FILTER_SIZE, 4*FILTER_NUM, 0.2, batch_norm)
    pool_16 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_32)

    conv_16 = fmbconv(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling3D(pool_size=(2,2,2))(conv_16)

    conv_8 = fmbconv(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # conv_8 = AtrousSpatialPyramidWaterFallPool(conv_8, useLeakyReLU=True)

    up_16 = Conv3DTranspose(8*FILTER_NUM, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format="channels_last")(conv_8)

    g1 = gating_signal(conv_8, 8*FILTER_NUM)
    a1 = attention_block(conv_16, g1, 8*FILTER_NUM)

    up_16 = layers.concatenate([up_16, a1], axis=4)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    up_32 = Conv3DTranspose(4*FILTER_NUM, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format="channels_last")(up_conv_16)


    g2 = gating_signal(up_conv_16, 4*FILTER_NUM)
    a2 = attention_block(conv_32, g2, 4*FILTER_NUM)

    up_32 = layers.concatenate([up_32, a2], axis=4)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, 0.2, batch_norm)

    up_64 = Conv3DTranspose(2*FILTER_NUM, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format="channels_last")(up_conv_32)

    g3 = gating_signal(up_conv_32, 2*FILTER_NUM)
    a3 = attention_block(conv_64, g3, 2*FILTER_NUM)

    up_64 = layers.concatenate([up_64, a3], axis=4)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, 0.2, batch_norm)

    up_128 = Conv3DTranspose(FILTER_NUM, (2, 2, 2), strides=(2, 2, 2), padding='same', data_format="channels_last")(up_conv_64)

    g4 = gating_signal(up_conv_64, 1*FILTER_NUM)
    a4 = attention_block(conv_128, g4, 1*FILTER_NUM)

    up_128 = layers.concatenate([up_128, a4], axis=4)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, 0, batch_norm)

   
    conv_final = layers.Conv3D(num_classes, kernel_size=(1,1,1))(up_conv_128)


    conv_final = layers.Activation('softmax', dtype=mixed_precision.Policy('float32'))(conv_final)  # Float32 for using Mixed Precision

    model = models.Model(inputs, conv_final, name="E-AG-CBAM")
    return model




LR = 0.0001
optim = Adam(LR)

import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = total_loss = sm.losses.categorical_focal_dice_loss

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


media_start = './'


from keras.models import load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler

filepath= media_start+"{epoch:02d}.hdf5"

checkpoint = ModelCheckpoint(filepath, save_freq="epoch",save_best_only=False,
    save_weights_only=False)

csv_logger = CSVLogger(media_start+"log.csv")


model = _model(IMG_HEIGHT=128, IMG_WIDTH=128,  IMG_DEPTH=128,  IMG_CHANNELS=3,  num_classes=4)

model.summary()
model.compile(optimizer = optim, loss=total_loss, metrics=metrics)






