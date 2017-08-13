import numpy as np
import os
from scipy import misc
import tensorflow as tf

from utils import data_generator_with_attack

class Evaluator(object):
    def __init__(self):
        pass

    @staticmethod
    def evaluate(sess, input_tensor, output, ground_truth_csv,
                 batch_size, orig_img_dir, attacked_img_dir, output_dir, output_file):
        """Evaluate attacked images

        :param sess: tensorflow session
        :param input_tensor: model input placeholder
        :param output: model output tf.op
        :param ground_truth_csv: csv file with keys ['ImageId', 'PredictedLabel']
        :param batch_size: mini-batch size
        :param orig_img_dir: original image directory
        :param attacked_img_dir: attacked image directory
        :param output_dir: directory to store successed and failed images


        """
        batch_shape = [batch_size, input_tensor.shape[1], input_tensor.shape[2], 3]
        add_bg_class = (input_tensor.shape[1] == 224)

        orig_correct_num = 0.0
        after_attack_correct_num = 0.0
        after_attack_difference_num = 0.0
        no_effect_correct_num = 0.0
        total_num = 0.0

        if not os.path.exists(output_dir + 'Adversarial/FalseNegative'):
            os.makedirs(output_dir + 'Adversarial/FalseNegative')
        if not os.path.exists(output_dir + 'Adversarial/FalsePositive'):
            os.makedirs(output_dir + 'Adversarial/FalsePositive')
        if not os.path.exists(output_dir + 'Failed/'):
            os.makedirs(output_dir + 'Failed/')

        with tf.gfile.Open(output_file, 'w') as out_file:

            for img_file_paths, orig_images, attacked_images, gt_labels \
                    in data_generator_with_attack(ground_truth_csv, batch_shape, orig_img_dir, attacked_img_dir):

                orig_output_np = sess.run(output, feed_dict={input_tensor: orig_images})
                orig_output_np = np.argmax(orig_output_np, axis=1) + add_bg_class

                attacked_output_np = sess.run(output, feed_dict={input_tensor: attacked_images})
                attacked_output_np = np.argmax(attacked_output_np, axis=1) + add_bg_class

                orig_correctness = (orig_output_np == gt_labels)
                after_attack_correctness = (attacked_output_np == gt_labels)
                after_attack_difference = (attacked_output_np != orig_output_np)

                no_effect_correct_num += np.sum(np.multiply((attacked_output_np == orig_output_np), (orig_output_np == gt_labels)))

                orig_correct_num += np.sum(orig_correctness)
                after_attack_correct_num += np.sum(after_attack_correctness)
                after_attack_difference_num += np.sum(after_attack_difference)
                total_num += len(orig_output_np)

                print('')

                print('Attack Successed: {:5d} | Failed: {:5d}'.format(int(after_attack_difference_num),
                                                                       int(total_num - after_attack_difference_num)))
                print('{:<10}: {}'.format('origin', orig_output_np))
                print('{:<10}: {}'.format('attacked', attacked_output_np))
                print('=====================')

                attacked_images = attacked_images.astype(np.uint8)
                for i in range(len(after_attack_difference)):
                    out_file.write('{0},{1}\n'.format(img_file_paths[i], attacked_output_np[i]))
                    if after_attack_difference[i] == True:
                        if attacked_output_np[i] != gt_labels[i]:
                            misc.imsave(output_dir + 'Adversarial/FalseNegative/' + img_file_paths[i], attacked_images[i])
                        else:
                            misc.imsave(output_dir + 'Adversarial/FalsePositive/' + img_file_paths[i], attacked_images[i])
                    else:
                        misc.imsave(output_dir + 'Failed/' + img_file_paths[i], attacked_images[i])

        print('----------------------------------------')
        print('Orig Accuracy = %f' % (orig_correct_num / total_num))
        print('After Attack Accuracy = %f' % (after_attack_correct_num / total_num))
        print(
        'False positive attacks = %d' % (after_attack_correct_num - no_effect_correct_num))
