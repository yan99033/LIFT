# test.py ---
#
# Filename: test.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 17:16:20 2016 (+0100)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary: Dataset class for forming the data into a data_obj we
# can use with our learning framework.
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), EPFL Computer Vision Lab.

# Code:


from __future__ import print_function

import os

import cv2
import numpy as np

from Utils.custom_types import pathConfig
from Utils.dataset_tools.helper import load_patches
from Utils.dump_tools import loadh5
from Utils.kp_tools import IDX_ANGLE, loadKpListFromTxt2

number_of_process = 20


class data_obj(object):
    ''' Dataset Object class.

    Implementation of the dataset object
    '''

    def __init__(self, param, image_gray, kp_list, mean_std_dict):

        # Set parameters
        self.out_dim = 1        # a single regressor output (TODO)

        # Load data
        # self.x = None           # data (patches) to be used for learning [N,
        #                         # channel, w, h]
        # self.y = None           # label/target to be learned
        # self.ID = None          # id of the data for manifold regularization
        self.load_data(param, image_gray, kp_list)

        # Set parameters
        self.num_channel = self.x.shape[1]  # single channel image
        # patch width == patch height (28x28)
        self.patch_height = self.x.shape[2]
        self.patch_width = self.x.shape[3]
        self.mean_std_dict = mean_std_dict

    def load_data(self, param, image_gray, kp_list):

        print(' --------------------------------------------------- ')
        print(' Test Data Module ')
        print(' --------------------------------------------------- ')

        pathconf = pathConfig()
        pathconf.setupTrain(param, 0)

        cur_data = self.load_data_for_set(
            pathconf, param, image_gray, kp_list)
        self.x = cur_data[0]
        self.y = cur_data[1]
        self.ID = cur_data[2]
        self.pos = cur_data[3]
        self.angle = cur_data[4]
        self.coords = cur_data[5]

        print(' -- Loading finished')

    def load_data_for_set(self, pathconf, param,
                          image_gray, kp_list):

        bTestWithTestMeanStd = getattr(
            param.validation, 'bTestWithTestMeanStd', False)
        if bTestWithTestMeanStd:
            image_gray = image_gray - np.mean(image_gray)
            image_gray = image_gray + self.mean_std_dict['mean_x']

        # load the keypoint informations
        kp = np.asarray(kp_list)

        # grayscale image
        in_dim = 1

        # print('KP:')
        # print(kp)
        # print('KP size')
        # print(kp.shape)

        # Assign dummy values to y, ID, angle
        y = np.zeros((len(kp),))
        ID = np.zeros((len(kp),), dtype='int64')
        # angle = np.zeros((len(kp),))
        angle = np.pi / 180.0 * kp[:, IDX_ANGLE]  # store angle in radians

        # load patches with id (drop out of boundary)
        bPerturb = False
        fPerturbInfo = np.zeros((3,))
        dataset = load_patches(image_gray, kp, y, ID, angle, param.patch.fRatioScale,
                               param.patch.fMaxScale, param.patch.nPatchSize,
                               param.model.nDescInputSize, in_dim, bPerturb,
                               fPerturbInfo, bReturnCoords=True)

        x = dataset[0]
        y = dataset[1]
        ID = dataset[2]
        pos = dataset[3]
        angle = dataset[4]
        coords = dataset[5]

        return x, y, ID, pos, angle, coords


#
# test.py ends here
