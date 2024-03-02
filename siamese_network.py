# import the necessary packages
import tensorflow
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout,Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D,Flatten,Reshape, Multiply, Add, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D,Concatenate, UpSampling2D, Activation
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet121

from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam
import numpy as np

def attention_block(inputs, reduction_ratio=8):
    
    ##ResNet50
    #avg_pool = GlobalAveragePooling2D()(inputs)
    # Reshape for spatial attention
    #spatial_attention_map = Reshape((1, 1, -1))(avg_pool)
    # Apply 1x1 convolution to create attention map
    #spatial_attention_map = Conv2D(1, kernel_size=1, activation='sigmoid')(spatial_attention_map)
    # Multiply attention map with input tensor
    #attention_applied = Multiply()([inputs, spatial_attention_map])
    #x = Conv2D(filters=128, kernel_size=3, padding='same')(attention_applied)
    #x = Activation('relu')(x)    
    
    ##VGG16
    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, -1))(x)
    x = Dense(inputs.shape[-1] // 2, activation='relu')(inputs)
    x = Dense(inputs.shape[-1]  , activation='sigmoid')(x)
    x = Multiply()([inputs, x])

    ##DenseNet121 Attention Mechanism
    #attention_weights = Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(inputs)
    # Apply attention to the input features
    #x = Multiply()([inputs, attention_weights])
    #conv_features = Conv2D(filters=512, kernel_size=3, padding='same')(x)
    #conv_features = BatchNormalization()(conv_features)
    #conv_features = Activation('relu')(conv_features)
    
    # Apply another attention mechanism to the conv_features
    #attention_weights_2 = Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(conv_features)
    #x = Multiply()([conv_features, attention_weights_2])
    #conv_features = Conv2D(filters=128, kernel_size=1, padding='same')(x)
    #conv_features = BatchNormalization()(conv_features)
    #x = Activation('relu')(conv_features)
    
    return x


def build_siamese_model(inputShape, embeddingDim=64):
    #input_shape = (512, 512, 3)
    left_input = Input(inputShape)
    #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=inputShape)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=inputShape)
    #base_model=InceptionV3(include_top=False,weights="imagenet",input_tensor=None,input_shape=(512,512,3),pooling=None,classes=1000)

    for layer in base_model.layers[:-2]:
        layer.trainable = False

    x = base_model(left_input)
    x = attention_block(x, reduction_ratio=8)
    x = GlobalAveragePooling2D()(x)


    output = Dense(embeddingDim)(x)
    
    model = Model(left_input,[output])
    
    print(model.summary())
    return model