# ghostnet_model.py
import tensorflow as tf
from tensorflow.keras import layers

def make_ghost_module(inputs, out_channels, ratio=2, conv_kernel=1, dw_kernel=3, stride=1, use_relu=True, name=None):
    init_channels = int(out_channels / ratio)
    new_channels = out_channels - init_channels

    x = layers.Conv2D(init_channels, conv_kernel, strides=stride, padding='same', use_bias=False, name=name+'_conv1')(inputs)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    if use_relu:
        x = layers.ReLU(name=name+'_relu1')(x)

    ghost = layers.DepthwiseConv2D(dw_kernel, strides=1, padding='same', use_bias=False, name=name+'_dconv')(x)
    ghost = layers.BatchNormalization(name=name+'_bn2')(ghost)
    if use_relu:
        ghost = layers.ReLU(name=name+'_relu2')(ghost)

    out = layers.Concatenate(name=name+'_concat')([x, ghost])
    return out

def build_ghostnet_feature_extractor(input_tensor):
    x = make_ghost_module(input_tensor, 16, name='g1')
    x = make_ghost_module(x, 32, stride=2, name='g2')
    x = make_ghost_module(x, 64, stride=2, name='g3')
    x = make_ghost_module(x, 128, stride=2, name='g4')
    x = make_ghost_module(x, 256, stride=2, name='g5')

    x = layers.GlobalAveragePooling2D(name='ghost_global_pool')(x)
    return x
