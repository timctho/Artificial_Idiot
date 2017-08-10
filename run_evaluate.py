from evaluator import Evaluator
import tensorflow as tf
import defense_nets

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_file',
                       'inception_v3.ckpt',
                       'Model which used for defense.')

tf.flags.DEFINE_string('ground_truth_csv',
                       'images.csv',
                       'CSV file with keys [ImageId, TrueLabel].')

tf.flags.DEFINE_string('orig_img_dir',
                       'Devset_imgs/',
                       'Directory of original images.')

tf.flags.DEFINE_string('attacked_img_dir',
                       'Attacked_imgs/',
                       'Directory of attacked images.')

tf.flags.DEFINE_string('output_dir',
                       'Evaluated_outputs/',
                       'Directory to store evaluated outputs.')

tf.flags.DEFINE_string('output_file',
                       'output.txt',
                       'File to store evaluated outputs.')

tf.flags.DEFINE_integer('batch_size',
                        64,
                        'Mini-batch size.')


FLAGS = tf.flags.FLAGS


def main(argv):
    if FLAGS.model_file.startswith('inception_v3'):
        input_tensor, output_op = defense_nets.Inception_v3().get_in_out()
    elif FLAGS.model_file.startswith('resnet_v2_50'):
        input_tensor, output_op = defense_nets.Resnet_v2_50().get_in_out()
    elif FLAGS.model_file.startswith('vgg_19'):
        input_tensor, output_op = defense_nets.VGG19().get_in_out()
    else:
        raise Exception('Unsupported model!')

    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.model_file)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        Evaluator.evaluate(sess=sess,
                           input_tensor=input_tensor,
                           output=output_op,
                           ground_truth_csv=FLAGS.ground_truth_csv,
                           batch_size=FLAGS.batch_size,
                           orig_img_dir=FLAGS.orig_img_dir,
                           attacked_img_dir=FLAGS.attacked_img_dir,
                           output_dir=FLAGS.output_dir,
                           output_file=FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
