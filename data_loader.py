# -*- coding: utf-8 -*-
# """
# mvtec_ad_loader.py
#   2021.05.06. @chanwoo.park
#   Load mvtec_ad dataset
#   Reference:
#       Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
#       "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
#       IEEE Conference on Computer Vision and Pattern Recognition, 2019
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os

# 2. Third-party modules
import cv2
import numpy as np
import tensorflow as tf

# 3. Own modules


###########
#   CLASS #
###########
class MVTecADLoader(object):
    base_path = r'D:\mvtec_ad'

    train, test = None, None
    num_train, num_test = 0, 0

    category = {'bottle': ['good', 'broken_large', 'broken_small', 'contamination'],
                'cable': ['good', 'bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation',
                          'missing_cable', 'missing_wire', 'poke_insulation'],
                'capsule': ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
                'carpet': ['good', 'color', 'cut', 'hole', 'metal_contamination', 'thread'],
                'grid': ['good', 'bent', 'broken', 'glue', 'metal_contamination', 'thread'],
                'hazelnut': ['good', 'crack', 'cut', 'hole', 'print'],
                'leather': ['good', 'color', 'cut', 'fold', 'glue', 'poke'],
                'metal_nut': ['good', 'bent', 'color', 'flip', 'scratch'],
                'pill': ['good', 'color', 'combined', 'contamination', 'crack', 'faulty_imprint', 'pill_type',
                         'scratch'],
                'screw': ['good', 'manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top'],
                'tile': ['good', 'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
                'toothbrush': ['good', 'defective'],
                'transistor': ['good', 'bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
                'wood': ['good', 'color', 'combined', 'good', 'hole', 'liquid', 'scratch'],
                'zipper': ['good', 'broken_teeth', 'combined', 'fabric_border', 'fabric_interior', 'rough',
                           'split_teeth', 'squeezed_teeth']}

    def setup_base_path(self, path):
        self.base_path = path

    def load(self, category, repeat=4, max_rot=10):
        # data, mask, binary anomaly label (0 for anomaly, 1 for good)
        x, y, z = [], [], []

        # Load train set
        path = os.path.join(os.path.join(self.base_path, category), 'train/good')
        files = os.listdir(path)

        zero_mask = tf.zeros(shape=(224, 224), dtype=tf.int32)

        for rdx in range(repeat):
            for _files in files:
                full_path = os.path.join(path, _files)
                img = self._read_image(full_path=full_path)

                if not max_rot == 0:
                    img = tf.keras.preprocessing.image.random_rotation(img, max_rot)

                mask = zero_mask

                x.append(img)
                y.append(mask)
                z.append(1)

        x = np.asarray(x)
        y = np.asarray(y)
        self.num_train = len(x)

        x = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x, dtype=tf.float32))
        y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y, dtype=tf.int32))
        z = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(z, dtype=tf.int32))

        self.train = tf.data.Dataset.zip((x, y, z))

        # data, anomaly label (e.g., good, cut, ..., etc.), binary anomaly label (0 for anomaly, 1 for good)
        x, y, z = [], [], []

        # Load test set
        for _label in self.category[category]:
            path = os.path.join(os.path.join(self.base_path, category), 'test/{}'.format(_label))

            files = os.listdir(path)
            for _files in files:
                full_path = os.path.join(path, _files)
                img = self._read_image(full_path=full_path)

                if _label == 'good':
                    mask = zero_mask
                else:
                    mask_path = os.path.join(os.path.join(self.base_path, category), 'ground_truth/{}'.format(_label))
                    _mask_path = os.path.join(mask_path, '{}_mask.png'.format(_files.split('.')[0]))
                    mask = cv2.resize(cv2.imread(_mask_path, flags=cv2.IMREAD_GRAYSCALE), dsize=(256, 256)) / 255
                    mask = mask[16:-16, 16:-16]
                    mask = tf.convert_to_tensor(mask, dtype=tf.int32)

                x.append(img)
                y.append(mask)
                z.append(int(self.category[category].index(_label) == 0))

        x = np.asarray(x)
        y = np.asarray(y)
        self.num_test = len(x)

        x = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x, dtype=tf.float32))
        y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y, dtype=tf.int32))
        z = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(z, dtype=tf.int32))

        self.test = tf.data.Dataset.zip((x, y, z))

    @staticmethod
    def _read_image(full_path, flags=cv2.IMREAD_COLOR):
        img = cv2.imread(full_path, flags=flags)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        img = cv2.resize(img, dsize=(256, 256))

        img = img[16:-16, 16:-16, :]

        return img
