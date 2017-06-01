# From LIFT: Learned Invariant Feature Transform (https://github.com/cvlab-epfl/LIFT)
# Original file: compute_detector.py

# Author: Shing-Yan Loo (University of Alberta / Universiti Putra Malaysia)
# Created: 12:55pm, May 27, 2017

from Utils.custom_types import paramGroup, paramStruct, pathConfig
from Utils.dataset_tools import test2 as data_module
#from Utils.sift_tools import recomputeOrientation
from Utils.kp_tools import IDX_ANGLE, update_affine, XYZS2kpList, get_XYZS_from_res_list, saveKpListToTxt
from Utils.solvers import TestImage
from Utils.solvers import Test
from Utils.dump_tools import saveh5

import cv2
import h5py
import numpy as np
import os

from copy import deepcopy

# floatX = 'float32'
bUseTheano = True

class Detector:
    def __init__(self, config_file, model_dir, num_keypoint, image_gray, b_save_png, mean_std_dict):
        # ------------------------------------------------------------------------
        # Setup and load parameters
        self.param = paramStruct()
        self.param.loadParam(config_file, verbose=False)
        self.param_desc = paramStruct()
        self.param_desc.loadParam(config_file, verbose=False)
        self.pathconf = pathConfig()
        self.b_draw = b_save_png
        self.num_keypoint = num_keypoint
        self.mean_std_dict = mean_std_dict

        self.param_desc = paramStruct()
        self.param_desc.loadParam(config_file, verbose=False)

        # Initialize self.pathconf structure
        self.pathconf.setupTrain(self.param, 0)

        # Overwrite with hard-coded base model
        setattr(self.param_desc.model, "descriptor_export_folder",
                os.getenv("_LIFT_BASE_PATH", "") + "/models/base")

        # add the mean and std of the learned model to the param
        self.param_desc.online = paramGroup()
        setattr(self.param_desc.online, 'mean_x', mean_std_dict['mean_x'])
        setattr(self.param_desc.online, 'std_x', mean_std_dict['std_x'])

        self.param.online = paramGroup()
        setattr(self.param.online, 'mean_x', self.mean_std_dict['mean_x'])
        setattr(self.param.online, 'std_x', self.mean_std_dict['std_x'])

        # Use model dir if given
        if model_dir is not None:
            self.pathconf.result = model_dir

        self.new_kp_list = None
        self.test_data_in = None

        # initialize scales and sizes
        # check size
        self.image_height = image_gray.shape[0]
        self.image_width = image_gray.shape[1]

        # Multiscale Testing
        self.scl_intv = getattr(self.param.validation, 'nScaleInterval', 4)
        # min_scale_log2 = 1  # min scale = 2
        # max_scale_log2 = 4  # max scale = 16
        min_scale_log2 = getattr(self.param.validation, 'min_scale_log2', 1)
        max_scale_log2 = getattr(self.param.validation, 'max_scale_log2', 4)
        # Test starting with double scale if small image
        min_hw = np.min(image_gray.shape[:2])
        if min_hw <= 1600:
            print("INFO: Testing double scale")
            min_scale_log2 -= 1
        # range of scales to check
        num_division = (max_scale_log2 - min_scale_log2) * (self.scl_intv + 1) + 1
        self.scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                        num_division)

        # convert scale to image resizes
        self.resize_to_test = ((float(self.param.model.nPatchSizeKp - 1) / 2.0) /
                          (self.param.patch.fRatioScale * self.scales_to_test))

        # check if resize is valid
        min_hw_after_resize = self.resize_to_test * np.min(image_gray.shape[:2])
        is_resize_valid = min_hw_after_resize > self.param.model.nFilterSize + 1

        # if there are invalid scales and resizes
        if not np.prod(is_resize_valid):
            # find first invalid
            first_invalid = np.where(True - is_resize_valid)[0][0]

            # remove scales from testing
            self.scales_to_test = self.scales_to_test[:first_invalid]
            self.resize_to_test = resize_to_test[:first_invalid]

        # print('resize to test is {}'.format(resize_to_test))
        # print('scales to test is {}'.format(scales_to_test))

        # NMS
        self.nearby = int(np.round(
            (0.5 * (self.param.model.nPatchSizeKp - 1.0) *
             float(self.param.model.nDescInputSize) /
             float(self.param.patch.nPatchSize))
        ))
        fNearbyRatio = getattr(self.param.validation, 'fNearbyRatio', 1.0)
        # Multiply by quarter to compensate
        fNearbyRatio *= 0.25
        self.nearby = int(np.round(self.nearby * fNearbyRatio))
        self.nearby = max(self.nearby, 1)

        self.nms_intv = getattr(self.param.validation, 'nNMSInterval', 2)
        self.edge_th = getattr(self.param.validation, 'fEdgeThreshold', 10)
        self.do_interpolation = getattr(self.param.validation, 'bInterpolate', True)

        self.fScaleEdgeness = getattr(self.param.validation, 'fScaleEdgeness', 0)

        # Create a list of param for different scales
        self.resize_to_test_list = []
        # Copy current param settings for detector
        self.param_detect = deepcopy(self.param)
        for resize in self.resize_to_test:
            # Just designate only one scale to bypass resizing. Just a single
            # number is fine, no need for a specific number
            param_cur_scale = deepcopy(self.param_detect)
            param_cur_scale.patch.fScaleList = [1.0]

            # Load the mean and std
            param_cur_scale.online = paramGroup()
            setattr(param_cur_scale.online, 'mean_x', self.mean_std_dict['mean_x'])
            setattr(param_cur_scale.online, 'std_x', self.mean_std_dict['std_x'])

            # disable orientation to avoid complications
            param_cur_scale.model.sOrientation = 'bypass'
            # disable descriptor to avoid complications
            param_cur_scale.model.sDescriptor = 'bypass'

            self.resize_to_test_list.append(param_cur_scale)


        # -------------------------------------------------------------------------
        # Modify the network so that we bypass the keypoint part and the
        # descriptor part (for orientation detection).
        self.param.model.sDetector = 'bypass'
        # This ensures that you don't create unecessary scale space
        self.param.model.fScaleList = np.array([1.0])
        self.param.patch.fMaxScale = np.max(self.param.model.fScaleList)
        # this ensures that you don't over eliminate features at boundaries
        self.param.model.nPatchSize = int(np.round(self.param.model.nDescInputSize) *
                                     np.sqrt(2))
        self.param.patch.fRatioScale = (float(self.param.patch.nPatchSize) /
                                   float(self.param.model.nDescInputSize)) * 6.0
        self.param.model.sDescriptor = 'bypass'

        # -------------------------------------------------------------------------
        # Modify the network so that we bypass the keypoint part and the
        # orientation part (for keypoint descriptor)
        self.param_desc.model.sDetector = 'bypass'
        # This ensures that you don't create unecessary scale space
        self.param_desc.model.fScaleList = np.array([1.0])
        self.param_desc.patch.fMaxScale = np.max(self.param.model.fScaleList)
        # this ensures that you don't over eliminate features at boundaries
        self.param_desc.model.nPatchSize = int(np.round(self.param.model.nDescInputSize) *
                                     np.sqrt(2))
        self.param_desc.patch.fRatioScale = (float(self.param.patch.nPatchSize) /
                                   float(self.param.model.nDescInputSize)) * 6.0
        self.param_desc.model.sOrientation = 'bypass'

        # -------------------------------------------------------------------------
        # store descriptors and keypoints for matching


        # ------------------------------------------------------------------------

    def detect_keypoint(self, image_gray, image_color):
        # Run for each scale
        test_res_list = []
        print('resize_to_test..............................................')
        print(self.resize_to_test)
        for resize, param in zip(self.resize_to_test, self.resize_to_test_list):
            # resize according to how we extracted patches when training
            new_height = np.cast['int'](np.round(self.image_height * resize))
            new_width = np.cast['int'](np.round(self.image_width * resize))
            image = cv2.resize(image_gray, (new_width, new_height))

            # run test
            if bUseTheano:
                # turn back verbose on
                test_res, _ = TestImage(self.pathconf, param,
                                                   image, verbose=False)
                test_res = np.squeeze(test_res)

            else:
                sKpNonlinearity = getattr(self.param.model, 'sKpNonlinearity', 'None')
                test_res = apply_learned_filter_2_image_no_theano(
                    image, self.pathconf.result,
                    self.param.model.bNormalizeInput,
                    sKpNonlinearity,
                    verbose=False)


            # pad and add to list
            test_res_list += [np.pad(test_res,
                                     int((self.param.model.nFilterSize - 1) / 2),
                                     # mode='edge')]
                                     mode='constant',
                                     constant_values=-np.inf)]

        # ------------------------------------------------------------------------
        # Non-max suppresion and draw.

        # The nonmax suppression implemented here is very very slow. COnsider this
        # as just a proof of concept implementation as of now.

        # Standard nearby : nonmax will check the approx. the same area as
        # descriptor support region.
        print("Performing NMS")
        res_list = test_res_list
        XYZS = get_XYZS_from_res_list(res_list, self.resize_to_test,
                                      self.scales_to_test, self.nearby, self.edge_th,
                                      self.scl_intv, self.nms_intv, self.do_interpolation,
                                      self.fScaleEdgeness)
        XYZS = XYZS[:self.num_keypoint]

        # ------------------------------------------------------------------------
        # Save as keypoint file to be used by the oxford thing
        print("Turning into kp_list")
        kp_list = XYZS2kpList(XYZS)  # note that this is already sorted

        # ------------------------------------------------------------------------
        # Also compute angles with the SIFT method, since the keypoint component
        # alone has no orientations.
        # print("Recomputing Orientations")
        # self.new_kp_list, _ = recomputeOrientation(image_gray, kp_list,
        #                                       bSingleOrientation=True)
        # print('KP_list')
        # print(kp_list)
        self.new_kp_list = kp_list

        # draw frame
        return self.draw_XYZS_to_img(image_color, XYZS)

    def compute_orientation(self, image_gray):
        # -------------------------------------------------------------------------
        # Load data in the test format
        self.test_data_in = data_module.data_obj(self.param, image_gray, self.new_kp_list, self.mean_std_dict)

        # -------------------------------------------------------------------------
        # Test using the test function
        _, oris, _ = Test(
            self.pathconf, self.param, self.test_data_in, test_mode="ori")

        # update keypoints and save as new
        kps = self.test_data_in.coords
        for idxkp in xrange(len(kps)):
            kps[idxkp][IDX_ANGLE] = oris[idxkp] * 180.0 / np.pi % 360.0
            kps[idxkp] = update_affine(kps[idxkp])

        # save as new keypoints
        self.new_kp_list = kps

    def compute_descriptor(self, image_gray, index):
        descs, _, _ = Test(
            self.pathconf, self.param_desc, self.test_data_in, test_mode="desc")

        save_dict = {}
        save_dict['keypoints'] = self.test_data_in.coords
        save_dict['descriptors'] = descs
        # print('Keypoints:')
        # print(self.test_data_in.coords)
        #
        # print('descriptors:')
        # for desc in descs:
        #     print(desc)
        output_file = str(index) + '.h5'
        saveh5(save_dict, output_file)

    # draw frames
    def draw_XYZS_to_img(self, image_color, XYZS):
        """ Drawing functino for displaying """
        # draw onto the original image
        # if cv2.__version__[0] == '3':
        #     linetype = cv2.LINE_AA
        # else:
        #     linetype = cv2.CV_AA
        [cv2.circle(image_color, tuple(np.round(pos).astype(int)),
                    np.round(rad * 6.0).astype(int), (0, 255, 0), 2,
                    lineType=cv2.LINE_AA)
         for pos, rad in zip(XYZS[:, :2], XYZS[:, 2])]

        return image_color

    # # draw matches
    # def draw_matches(kp1, kp2, desc1, desc2, img1, img2)
