import os
from utils import save_images
import tensorflow as tf
import defense_nets
from utils import data_generator
import attack_methods

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_file',
                       'vgg_19.ckpt',
                       'Model which used for defense.')

tf.flags.DEFINE_string('ground_truth_csv',
                       'images.csv',
                       'CSV file with keys [ImageId, TrueLabel].')

tf.flags.DEFINE_string('orig_img_dir',
                       'Devset_imgs/',
                       'Input directory with images.')

tf.flags.DEFINE_string('attacked_img_dir',
                       'Attacked_imgs',
                       'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon',
                      16.0,
                      'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('batch_size',
                        32,
                        'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def main(argv):

    if not os.path.exists(FLAGS.attacked_img_dir):
        os.mkdir(FLAGS.attacked_img_dir)

    with tf.Graph().as_default():

        if FLAGS.model_file.startswith('inception_v3'):
            normalize_image = True
            batch_shape = [FLAGS.batch_size, 299, 299, 3]
            model = defense_nets.Inception_v3()
        elif FLAGS.model_file.startswith('resnet_v2_50'):
            normalize_image = True
            batch_shape = [FLAGS.batch_size, 299, 299, 3]
            model = defense_nets.Resnet_v2_50()
        elif FLAGS.model_file.startswith('vgg_19'):
            normalize_image = False
            batch_shape = [FLAGS.batch_size, 224, 224, 3]
            model = defense_nets.VGG19()
        else:
            raise Exception('Unsupported model!')

        if normalize_image:
            eps = 2.0 * FLAGS.max_epsilon / 255.0
            x_adv = attack_methods.fgsm(model, eps, -1.0, 1.0)
        else:
            x_adv = attack_methods.fgsm(model, 16.0, -128.0, 128.0)


        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.model_file)

        img_count = 0
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images, _, orig_size_images in data_generator(FLAGS.ground_truth_csv,
                                                       batch_shape,
                                                       FLAGS.orig_img_dir):
                adv_images = sess.run(x_adv, feed_dict={model.input_tensor: images})

                img_count += len(adv_images)

                mean_perturbation = save_images(adv_images, orig_size_images, filenames, FLAGS.attacked_img_dir)
                print('Processed {:5d} images, mean perturbation= {}'.format(img_count, mean_perturbation))


if __name__ == '__main__':
    tf.app.run()
