# -*- coding: utf-8 -*-
# """
# padim.py
#   2021.05.02. @chanwoo.park
#   PaDiM algorithm
#   Reference:
#       Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
#       arXiv preprint arXiv:2011.08785 (2020).
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os

# 2. Third-party modules
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import morphology
from skimage.segmentation import mark_boundaries

# 3. Own modules


################
#   Definition #
################
def embedding_concat(l1, l2):
    bs, h1, w1, c1 = l1.shape
    _, h2, w2, c2 = l2.shape

    s = int(h1 / h2)
    x = tf.compat.v1.extract_image_patches(l1, ksizes=[1, s, s, 1], strides=[1, s, s, 1], rates=[1, 1, 1, 1],
                                           padding='VALID')
    x = tf.reshape(x, (bs, -1, h2, w2, c1))

    col_z = []
    for idx in range(x.shape[1]):
        col_z.append(tf.concat([x[:, idx, :, :, :], l2], axis=-1))
    z = tf.stack(col_z, axis=1)

    z = tf.reshape(z, (bs, h2, w2, -1))
    if s == 1:
        return z
    z = tf.nn.depth_to_space(z, block_size=s)

    return z


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i][0]

        gt = gts[i].transpose(1, 2, 0).squeeze()

        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0

        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img.astype(int))
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt.astype(int), cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img.astype(int), cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask.astype(int), cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img.astype(int))
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def draw_auc(fp_list, tp_list, auc, path):
    plt.figure()
    plt.plot(fp_list, tp_list, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path)

    plt.clf()
    plt.cla()
    plt.close()


def draw_precision_recall(precision, recall, base_line, path):
    f1_score = []
    for _idx in range(0, len(precision)):
        _precision = precision[_idx]
        _recall = recall[_idx]

        if _precision + _recall == 0:
            _f1 = 0
        else:
            _f1 = 2 * (_precision * _recall) / (_precision + _recall)
        f1_score.append(_f1)

    plt.figure()
    plt.plot(recall, precision, marker='.', label='precision-recall curve')
    plt.plot([0, 1], [base_line, base_line], linestyle='--', color='grey', label='No skill ({:.04f})'.format(base_line))
    plt.plot(recall, f1_score, linestyle='-', color='red', label='f1 score (Max.: {:.4f})'.format(np.max(f1_score)))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(path)

    plt.clf()
    plt.cla()
    plt.close()

    return np.max(f1_score)
