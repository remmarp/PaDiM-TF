# -*- coding: utf-8 -*-
# """
# main.py
#   2021.05.02. @chanwoo.park
#   run PaDiM algorithm
#   Reference:
#       Defard, Thomas, et al. "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
#       arXiv preprint arXiv:2011.08785 (2020).
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import argparse

# 2. Third-party modules
import random
import numpy as np
import tensorflow as tf

# 3. Own modules
from padim import padim

# For reproducibility, you can run scripts on CPU
# # Set CPU as available physical device
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
#
# # To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)

# For the reproducibility - please check https://github.com/NVIDIA/framework-determinism
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


################
#   Definition #
################
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int, help='What seed to use')
    parser.add_argument("--rd", default=1000, type=int, help='Random sampling dimension')
    parser.add_argument("--target", default='carpet', type=str, help="Which target to test")
    parser.add_argument("--batch_size", default=32, type=int, help="What batch size to use")
    parser.add_argument("--is_plot", default=True, type=bool, help="Whether to plot or not")
    parser.add_argument("--net", default='eff', type=str, help="Which embedding network to use", choices=['eff', 'res'])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = options()

    if opt.seed > -1:
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        tf.random.set_seed(opt.seed)

    padim(category=opt.target, batch_size=opt.batch_size, rd=opt.rd, net_type=opt.net, is_plot=opt.is_plot)
