from __future__ import division
from tensorflow.keras import layers,regularizers,constraints,initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda,Conv2D,LSTM, Concatenate,GlobalAveragePooling2D,Cropping2D

from tensorflow.keras.layers import Dropout, Activation,Dense,Reshape,Flatten
from tensorflow.keras.layers import Input,Add,multiply,concatenate
from tensorflow.keras.layers import Convolution2D, MaxPooling2D,UpSampling2D ,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K

import tensorflow as tf
import numpy as np





def kld(y_true, y_pred):

        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = y_pred
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        kld = K.sum(P * K.log(K.epsilon() + P/(eps + Q)), axis=[1, 2, 3])
#        print kld.shape     
        return kld


def TVdist(y_true, y_pred):
        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = y_pred
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        tv = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        return tv*0.5

def absTVdist(y_true, y_pred):
        P = y_true
#        print P.shape 
        P = P / (K.epsilon() + K.sum(K.abs(P), axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q =  y_pred
        Q = Q / (K.epsilon() + K.sum(K.abs(Q), axis=[1, 2, 3], keepdims=True))

        tv = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        return tv*0.5

def SoftTVdist(y_true, y_pred):
        P = K.exp(y_true)
#        print P.shape 
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
#        print P.shape 
        Q = K.exp(y_pred)
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        kld = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
#        print kld.shape     
        return tv*0.5
##################################


class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)


def diconv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              dilation_rate=(1, 1),
              activation='relu',
              use_bias=True,
              train_bn=None,              
              bnn= True,
              name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=use_bias,    #usually true
        name=conv_name)(x)
    if bnn:
#        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = BatchNorm(axis=bn_axis, scale=False, name=bn_name)(x, training=train_bn)
    x = Activation(activation, name=name)(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True,dilation_rate=(1, 1), train_bn=None):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), use_bias=use_bias, name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,dilation_rate=dilation_rate,
               padding='same',use_bias=use_bias,  name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),use_bias=use_bias,  name=conv_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x, training=train_bn)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, strides=(2, 2),dilation_rate=(1, 1), train_bn=None):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,use_bias=use_bias, 
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, dilation_rate=dilation_rate, padding='same',use_bias=use_bias, 
               name=conv_name_base + '2b')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=use_bias, name=conv_name_base + '2c')(x)
    #x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = BatchNorm(axis=bn_axis, name=bn_name_base + '2c')(x, training=train_bn)
    shortcut = Conv2D(filters3, (1, 1), strides=strides,use_bias=use_bias, 
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNorm(axis=bn_axis, name=bn_name_base + '1')(shortcut, training=train_bn)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

#keras resnet without bias.
def M_ResNet50nob(input_tensor=None, train_bn=None):
    input_shape = (None, None,3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # conv_1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2),padding='same')(x)
    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, use_bias=False,block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, use_bias=False,block='b', train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2,use_bias=False, block='c', train_bn=train_bn)
    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, use_bias=False,block='a', strides=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3,use_bias=False, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, use_bias=False,block='c', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3,use_bias=False, block='d', train_bn=train_bn)
    # conv_4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, use_bias=False,block='a',strides=(1, 1),dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,use_bias=False, block='b',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, use_bias=False,block='c',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, use_bias=False,block='d',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,use_bias=False, block='e',dilation_rate=(2, 2), train_bn=train_bn)
    x = identity_block(x, 3, [256, 256, 1024], stage=4,use_bias=False, block='f',dilation_rate=(2, 2), train_bn=train_bn)
    # conv_5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, use_bias=False,block='a',strides=(1, 1),dilation_rate=(4, 4), train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,use_bias=False, block='b',dilation_rate=(4, 4), train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5,use_bias=False, block='c',dilation_rate=(4, 4), train_bn=train_bn)

    model = Model(img_input, x)

    # Load weights
    model.load_weights('models/resnet50.h5')
    # for layer in model.layers:
    #     layer.training = False 
    return model         


def DINetEncoder(img_rows=480, img_cols=640,train_bn =None):
    inputimage = Input(shape=(img_rows, img_cols,3))
    base_model = M_ResNet50nob(input_tensor=inputimage, train_bn=train_bn)  # 
    
#    for layer in base_model.layers:
#        layer.trainable = False
    x5 = base_model.output

    x5 = Convolution2D(256, kernel_size=(1, 1), activation='relu',padding='same',dilation_rate=(1, 1),use_bias=False)(x5) 
    
    b1 = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',dilation_rate=(4, 4),use_bias=False)(x5) 
    
    b2 = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',dilation_rate=(8, 8),use_bias=False)(x5) 
    
    b3 = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',dilation_rate=(16, 16),use_bias=False)(x5) 

    x=layers.add([b1, b2,b3])  # 2618
    # x=layers.add([b1,b2]) 
    model = Model(inputs=[inputimage], outputs=[x,b1, b2,b3])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model 


def DINetDecoder(img_rows=480, img_cols=640):   
    inputimage = Input(shape=(int(img_rows/8), int(img_cols/8),256))
    x = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',use_bias=False)(inputimage) 
    x = Convolution2D(256, kernel_size=(3, 3), activation='relu',padding='same',use_bias=False)(x) 
    final_output = Convolution2D(1, kernel_size=(3, 3), activation='sigmoid',padding='same',use_bias=False)(x)

    final_output_up = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(final_output)
#    
    model = Model(inputs=[inputimage], outputs=[final_output_up])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model     


# train_bn=False better performance.
def DINet(img_rows=480, img_cols=640, train_bn=False):
    inputimage = Input(shape=(img_rows, img_cols,3))
    x,_,_,_  = DINetEncoder(img_rows, img_cols, train_bn)(inputimage)
    final_output_up = DINetDecoder(img_rows, img_cols)(x)
    model = Model(inputs=[inputimage], outputs=[final_output_up])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model


def DINetDecodernoas(img_rows=480, img_cols=640):   
    inputimage = Input(shape=(int(img_rows/8), int(img_cols/8),256))
    x = Convolution2D(256, kernel_size=(3, 3),padding='same',use_bias=False)(inputimage) 
    x = Convolution2D(256, kernel_size=(3, 3),padding='same',use_bias=False)(x) 

    final_output = Convolution2D(1, kernel_size=(3, 3),padding='same',use_bias=False)(x)   
    # avoid getting 'negative' saliency map.  It can be commented by using the absTVdist 
    final_output =  Lambda(lambda x:  (x- tf.reduce_min(x)))(final_output)  
    final_output =  Lambda(lambda x:  x/tf.reduce_max(x))(final_output)  

    final_output_up = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(final_output)  

#    
    model = Model(inputs=[inputimage], outputs=[final_output_up])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model 


def DINetvisual2o(img_rows=480, img_cols=640, train_bn=False):
    inputimage = Input(shape=(img_rows, img_cols,3))
    
#    for layer in base_model.layers:
#        layer.trainable = False
    x,b1,b2,b3 = DINetEncoder(img_rows, img_cols)(inputimage)
    # for layer in DINetEncoder.layers:
    #     layer.trainable = False
    final_output_up = DINetDecoder(img_rows, img_cols)(x)
    final_output_up2 = DINetDecodernoas(img_rows, img_cols)(x)  #for training  the whole model with two different decoders.
    # final_output_up2 = DINetDecodernoas(img_rows, img_cols)(b1)  #for visualizing b1
    model = Model(inputs=[inputimage], outputs=[final_output_up,final_output_up2])
    
#    for layer in model.layers:
#        print(layer.name, layer.input_shape, layer.output_shape)

    return model    