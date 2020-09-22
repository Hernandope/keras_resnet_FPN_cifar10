import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, applications


def get_backbone_ResNet50(input_shape):
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

def get_backbone_ResNet50V2(input_shape):
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50V2(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

def get_backbone_ResNet101(input_shape):
    """Builds ResNet101 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet101(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

class customFeaturePyramid2(keras.models.Model):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50, ResNet101 and V1 counterparts.
    """

    def __init__(self, backbone=None, class_number=10, **kwargs):
        super(customFeaturePyramid2, self).__init__(name="customFeaturePyramid2", **kwargs)
        self.backbone = backbone if backbone else get_backbone_ResNet50()
        self.class_number = class_number
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)
        self.dense_d1 = keras.layers.Dense(64,
                                           activation='relu',
                                           kernel_initializer='he_uniform')
        self.dense_d2 = keras.layers.Dense(self.class_number,
                                           activation='softmax',
                                           kernel_initializer='he_normal')

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        p3_output = keras.layers.Flatten()(p3_output)
        p4_output = keras.layers.Flatten()(p4_output)
        p5_output = keras.layers.Flatten()(p5_output)
        p6_output = keras.layers.Flatten()(p6_output)
        p7_output = keras.layers.Flatten()(p7_output)
        m1_output = keras.layers.Concatenate(axis=1)([p3_output,
                                                p4_output,
                                                p5_output,
                                                p6_output,
                                                p7_output])
        m1_output = keras.layers.Flatten()(m1_output)
        m1_output = self.dense_d1(m1_output)
        m1_output = self.dense_d2(m1_output)
        return m1_output
