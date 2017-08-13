import numpy as np
import pandas as pd
from scipy import misc
import tensorflow as tf

import os


def data_generator(ground_truth_csv, batch_shape, orig_img_dir):
    """Read png images from input directory in batches.

    :param ground_truth_csv: csv file with keys ['ImageId', 'PredictedLabel']
    :param batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    :param orig_img_dir: original image directory

    :return img_file_names: image names
    :return images: numpy array of normalized images
    :return labels: predicted image labels
    """

    idx = 0
    batch_size = batch_shape[0]
    orig_images = np.zeros(batch_shape)
    orig_size_images = np.zeros([batch_size, 299, 299, 3])
    gt_labels = np.zeros(batch_size)

    img_file_names = []

    csv_file = pd.read_csv(ground_truth_csv)

    for i in range(len(csv_file['ImageId'])):
        img_file_name = csv_file['ImageId'][i] + '.png'
        orig_image = misc.imread(orig_img_dir + img_file_name, mode='RGB').astype(np.float)

        orig_size_images[idx, :, :, :] = orig_image

        if orig_image.shape[0] != batch_shape[1] or orig_image.shape[1] != batch_shape[2]:
            orig_image = misc.imresize(orig_image, (batch_shape[1], batch_shape[2]))

        gt_label = csv_file['TrueLabel'][i]
        orig_images[idx, :, :, :] = orig_image
        gt_labels[idx] = gt_label
        img_file_names.append(img_file_name)

        idx += 1
        if idx == batch_size:
            yield img_file_names, orig_images, gt_labels, orig_size_images
            orig_images = np.zeros(batch_shape)
            orig_size_images = np.zeros([batch_size, 299, 299, 3])
            gt_labels = np.zeros(batch_size)
            img_file_names = []
            idx = 0
    if idx > 0:
        yield img_file_names[0:idx], orig_images[0:idx, ...], gt_labels[0:idx], orig_size_images[0:idx, ...]


def data_generator_with_attack(ground_truth_csv, batch_shape, orig_img_dir, attacked_img_dir):
    """Read png images from input directory in batches.

    :param ground_truth_csv: csv file with keys ['ImageId', 'TrueLabel']
    :param batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    :param orig_img_dir: original image directory
    :param attacked_img_dir: attacked image directory

    :return img_file_names: image names
    :return images: numpy array of normalized images
    :return labels: predicted image labels
    """

    idx = 0
    batch_size = batch_shape[0]
    orig_images = np.zeros(batch_shape)
    attacked_images = np.zeros(batch_shape)
    gt_labels = np.zeros(batch_size)

    img_file_names = []

    csv_file = pd.read_csv(ground_truth_csv)

    for i in range(len(csv_file['ImageId'])):
        img_file_name = csv_file['ImageId'][i] + '.png'
        orig_image = misc.imread(orig_img_dir + img_file_name, mode='RGB').astype(np.float)
        attacked_image = misc.imread(attacked_img_dir + img_file_name, mode='RGB').astype(np.float)

        if orig_image.shape[0] != batch_shape[1] or orig_image.shape[1] != batch_shape[2]:
            orig_image = misc.imresize(orig_image, (batch_shape[1], batch_shape[2]))
            attacked_image = misc.imresize(attacked_image, (batch_shape[1], batch_shape[2]))

        gt_label = csv_file['TrueLabel'][i]
        orig_images[idx, :, :, :] = orig_image
        attacked_images[idx, :, :, :] = attacked_image
        gt_labels[idx] = gt_label
        img_file_names.append(img_file_name)

        idx += 1
        if idx == batch_size:
            yield img_file_names, orig_images, attacked_images, gt_labels
            orig_images = np.zeros(batch_shape)
            attacked_images = np.zeros(batch_shape)

            gt_labels = np.zeros(batch_size)
            img_file_names = []
            idx = 0
    if idx > 0:
        yield img_file_names[0:idx], orig_images[0:idx, ...], attacked_images[0:idx, ...], gt_labels[0:idx]


def save_images(adv_images, orig_images, filenames, output_dir):
    """Saves images to the output directory.

    :param adv_images: array with minibatch of images
    :param filenames: list of filenames without path
    :param output_dir: directory where to save images
    """

    upper_bound = np.clip(orig_images+16, 0, 255)
    lower_bound = np.clip(orig_images-16, 0, 255)

    perturbation = 0.0
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            if adv_images.shape[1] == 299:
                img = (((adv_images[i, :, :, :] + 1.0) * 0.5) * 255.0)
                clipped_img = np.clip(img, lower_bound[i], upper_bound[i])
                perturbation += np.mean(np.abs(clipped_img-orig_images[i]))
            else:
                img = (adv_images[i, :, :, :] + [123.68, 116.78, 103.94])
                img = misc.imresize(img, (299, 299))
                clipped_img = np.clip(img, lower_bound[i], upper_bound[i])
                perturbation += np.mean(np.abs(clipped_img-orig_images[i]))

            misc.imsave(f, clipped_img.astype(np.uint8))
    return perturbation / adv_images.shape[0]