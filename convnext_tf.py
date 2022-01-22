import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, initializers


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It is also referred tp as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Padding(layers.Layer):
    def __init__(self, kernel_size: int, **kwargs):
        super().__init__(**kwargs)
        padding = (
            max(kernel_size) // 2
            if isinstance(kernel_size, (list, tuple))
            else kernel_size // 2
        )
        self.padding_layer = layers.ZeroPadding2D(padding=padding)

    def call(self, inputs, training=None):
        return self.padding_layer(inputs)


class Block(tf.keras.Model):
    """ConvNeXt block.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        self.dw_conv_1_padding = Padding(kernel_size=7)
        self.dw_conv_1 = layers.Conv2D(
            filters=dim,
            kernel_size=7,
            padding="valid",
            groups=dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=initializers.Zeros(),
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv_1 = layers.Dense(
            4 * dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=initializers.Zeros(),
        )
        self.act_fn = layers.Activation("gelu")
        self.pw_conv_2 = layers.Dense(
            dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=initializers.Zeros(),
        )
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(self.dw_conv_1_padding(x))
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return inputs + self.drop_path(x)


def get_convnext_model(
    model_name="convnext_tiny_1k",
    input_shape=(224, 224, 3),
    num_classes=1000,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
):
    """Implements ConvNeXt family of models given a configuration.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Note: `predict()` fails on CPUs because of group convolutions. The fix is recent at
    the time of the development: https://github.com/keras-team/keras/pull/15868. It's
    recommended to use a GPU / TPU.
    """

    inputs = layers.Input(input_shape)
    stem = keras.Sequential(
        [
            layers.Conv2D(
                dims[0],
                kernel_size=4,
                strides=4,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                bias_initializer=initializers.Zeros(),
            ),
            layers.LayerNormalization(epsilon=1e-6),
        ],
        name="stem",
    )

    downsample_layers = []
    downsample_layers.append(stem)
    for i in range(3):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(
                    dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                    bias_initializer=initializers.Zeros(),
                ),
            ],
            name=f"downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)

    stages = []
    dp_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(4):
        stage = keras.Sequential(
            [
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        name=f"convnext_block_{i}_{j}",
                    )
                    for j in range(depths[i])
                ]
            ],
            name=f"convnext_stage_{i}",
        )
        stages.append(stage)
        cur += depths[i]

    x = inputs
    for i in range(len(stages)):
        x = downsample_layers[i](x)
        x = stages[i](x)

    x = layers.GlobalAvgPool2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    outputs = layers.Dense(
        num_classes,
        name="classification_head",
        kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        bias_initializer=initializers.Zeros(),
    )(x)

    return keras.Model(inputs, outputs, name=model_name)
