import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, vgg, resnet_v2

slim = tf.contrib.slim


class Inception_v3():
    def __init__(self):
        batch_shape = [None, 299, 299, 3]
        self.input_tensor = tf.placeholder(tf.float32, shape=batch_shape)

        prepro_input_tensor = tf.div(self.input_tensor, 255.0)
        prepro_input_tensor = tf.multiply(prepro_input_tensor, 2.0)
        self.prepro_input_tensor = tf.subtract(prepro_input_tensor, 1.0)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(inputs=self.prepro_input_tensor,
                                                   num_classes=1001,
                                                   is_training=False,
                                                   reuse=None)
        output = end_points['Predictions']
        self.probs = output.op.inputs[0]

    def __call__(self, *args, **kwargs):
        return self.probs

    def get_in_out(self):
        return self.input_tensor, self.probs


class Resnet_v2_50():
    def __init__(self):
        batch_shape = [None, 299, 299, 3]
        self.input_tensor = tf.placeholder(tf.float32, shape=batch_shape)

        prepro_input_tensor = tf.div(self.input_tensor, 255.0)
        prepro_input_tensor = tf.multiply(prepro_input_tensor, 2.0)
        self.prepro_input_tensor = tf.subtract(prepro_input_tensor, 1.0)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            _, end_points = resnet_v2.resnet_v2_50(self.prepro_input_tensor,
                                                   num_classes=1001,
                                                   is_training=False)
        output = end_points['predictions']
        self.probs = output.op.inputs[0]

    def __call__(self, *args, **kwargs):
        return self.probs

    def get_in_out(self):
        return self.input_tensor, self.probs


class VGG19():
    def __init__(self):
        batch_shape = [None, 224, 224, 3]
        self.input_tensor = tf.placeholder(tf.float32, shape=batch_shape)
        self.prepro_input_tensor = self.input_tensor - [123.68, 116.78, 103.94]

        self.logits, _ = vgg.vgg_19(inputs=self.prepro_input_tensor,
                                    num_classes=1000,
                                    is_training=False)

    def __call__(self, *args, **kwargs):
        return self.logits

    def get_in_out(self):
        return self.input_tensor, self.logits
