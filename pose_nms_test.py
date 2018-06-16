# -*- coding: utf-8 -*-
"""
@author: fangyh09
"""

import unittest

import numpy.testing as nptest

from pose_nms import *


class TestPoseNMS(unittest.TestCase):

    def test_get_keypoint_width(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        pose_loc = np.ones([17, 2])
        pose_loc[0][0] = 7
        pose_loc[0][1] = 3
        nptest.assert_almost_equal(get_keypoint_width(pose_loc), 0.6,
                                   decimal=2)

    def test_get_pose_dis1(self):
        # def get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width,
        #                  delta1, delta2, mu):
        #     """
        #     <class 'numpy'>
        #     :param choose_idx:
        #     :param pose_conf: [n x 17]
        #     :param pose_loc: [n x 17 x 2]
        #     """
        #     pass
        choose_idx = 0
        pose_conf = np.zeros([2, 17])
        pose_loc = np.ones([2, 17, 2])
        keypoint_width = 10
        delta1 = 1
        delta2 = 1
        mu = 1
        nptest.assert_almost_equal(
            get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width,
                         delta1, delta2, mu), [17.0, 17.0], decimal=5)

    def test_get_pose_dis2(self):
        # def get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width,
        #                  delta1, delta2, mu):
        #     """
        #     <class 'numpy'>
        #     :param choose_idx:
        #     :param pose_conf: [n x 17]
        #     :param pose_loc: [n x 17 x 2]
        #     """
        #     pass
        choose_idx = 0
        pose_conf = np.ones([2, 17])
        pose_conf[0, 0] = 2
        pose_loc = np.zeros([2, 17, 2])
        pose_loc[1, 0, 1] = 11
        pose_loc[1, 1, 1] = 9
        keypoint_width = 10
        delta1 = 1
        delta2 = 1
        mu = 1
        nptest.assert_almost_equal(
            get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width,
                         delta1, delta2, mu), [27.20975971, 25.01985128],
            decimal=5)

    def test_merge_pose1(self):
        # def merge_pose(root_pose, cluster_pose_loc, cluster_pose_conf,
        #                keypoint_width):
        #     """
        #     :param root_pose: [17 x 2]
        #     :param cluster_pose_loc: [n x 17 x 2]
        #     :param cluster_pose_conf: [n x 17]
        #     :param keypoint_width: 0.1 x body_width
        #     :return:
        #     """
        #     pass
        root_pose = np.arange(34).reshape(-1, 2)
        cluster_pose_loc = np.ones([2, 17, 2])
        cluster_pose_loc[0] = root_pose + np.random.random([17, 2]) * 0.1
        cluster_pose_loc[1] = root_pose + np.random.random([17, 2]) * 0.1

        cluster_pose_conf = np.ones([2, 17])
        cluster_pose_conf[1] *= 9
        keypoint_width = 100
        nptest.assert_almost_equal(merge_pose(root_pose, cluster_pose_loc,
                                              cluster_pose_conf,
                                              keypoint_width)[0], root_pose,
                                   decimal=1)

    def test_merge_pose2(self):
        # def merge_pose(root_pose, cluster_pose_loc, cluster_pose_conf,
        #                keypoint_width):
        #     """
        #     :param root_pose: [17 x 2]
        #     :param cluster_pose_loc: [n x 17 x 2]
        #     :param cluster_pose_conf: [n x 17]
        #     :param keypoint_width: 0.1 x body_width
        #     :return:
        #     """
        #     pass
        root_pose = np.arange(34).reshape(-1, 2)
        cluster_pose_loc = np.ones([2, 17, 2])
        cluster_pose_loc[0] = root_pose * 2
        cluster_pose_loc[1] = root_pose + np.random.random([17, 2]) * 0.1

        cluster_pose_conf = np.ones([2, 17])
        cluster_pose_conf[0, :] *= 9
        keypoint_width = 1000
        nptest.assert_almost_equal(merge_pose(root_pose, cluster_pose_loc,
                                              cluster_pose_conf,
                                              keypoint_width)[0], root_pose * 2,
                                   decimal=-1)

    def test_pck_match(self):
        # def PCK_match(choose_idx, pose_loc, keypoint_width):
        #     """
        #     :param choose_idx:
        #     :param pose_loc: [n x 17 x 2]
        #     :param keypoint_width: 0.1 x body_width
        #     :return:
        #     """
        #     pass
        choose_idx = 0
        pose_loc = np.zeros([3, 17, 2])
        pose_loc[1, 1, 0] = 12
        pose_loc[1, 2, 0] = 19
        pose_loc[2, 3, 0] = 13
        keypoint_width = 10
        nptest.assert_almost_equal(PCK_match(choose_idx, pose_loc,
                                             keypoint_width)[0],
                                   [17, 15, 16], decimal=3)
        nptest.assert_almost_equal(PCK_match(choose_idx, pose_loc,
                                             keypoint_width)[1],
                                   [5, 3, 4], decimal=3)

    def test_pose_nms1(self):
        # def pose_nms(detections, mu=1.7, delta1=1, delta2=2.65, gamma=22.48):
        #     """
        #     :param detections: shape: [1xnx(1+17+4+34)] <class 'numpy'>
        #     Example:
        #     """
        #     pass
        detections = np.zeros([1, 3, 56])
        # box conf
        detections[0, 0, 0] = 1
        detections[0, 1, 0] = 0.8
        detections[0, 2, 0] = 0.8

        pose_conf = detections[0, :, 1:18]
        box_conf = detections[0, :, 0]
        pose_loc = detections[0, :, 22:]
        box_loc = detections[0, :, 18:22]

        # keypoint conf
        detections[0, :, 1:18] += 1

        # keypoint loc
        detections[0, 0, 22:] += np.arange(34)
        detections[0, 1, 22:] += np.arange(34) + np.random.random() * 0.1
        detections[0, 2, 22:] += np.arange(34) * np.random.random() * 0.01

        nptest.assert_almost_equal(pose_nms(detections), np.array([[
            [1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1.,
             0., 0., 0., 0., 0.0430685, 1.0430685,
             2.0430685, 3.0430685, 4.0430685, 5.0430685, 6.0430685, 7.0430685,
             8.0430685, 9.0430685, 10.0430685, 11.0430685, 12.0430685,
             13.0430685,
             14.0430685, 15.0430685, 16.0430685, 17.0430685, 18.0430685,
             19.0430685,
             20.0430685, 21.0430685, 22.0430685, 23.0430685, 24.0430685,
             25.0430685,
             26.0430685, 27.0430685, 28.0430685, 29.0430685, 30.0430685,
             31.0430685,
             32.0430685, 33.0430685]]]), decimal=1)

    def test_pose_nms2(self):
        # def pose_nms(detections, mu=1.7, delta1=1, delta2=2.65, gamma=22.48):
        #     """
        #     :param detections: shape: [1xnx(1+17+4+34)] <class 'numpy'>
        #     Example:
        #     """
        #     pass
        detections = np.zeros([1, 3, 56])
        # box conf
        detections[0, 0, 0] = 1
        detections[0, 1, 0] = 0.8
        detections[0, 2, 0] = 0.2

        pose_conf = detections[0, :, 1:18]
        box_conf = detections[0, :, 0]
        pose_loc = detections[0, :, 22:]
        box_loc = detections[0, :, 18:22]

        # keypoint conf
        detections[0, :, 1:18] += 1
        detections[0, 2, 1:18] -= 1

        # keypoint loc
        detections[0, 0, 22:] += np.arange(34)
        detections[0, 1, 22:] += np.arange(34) + np.random.random() * 0.1
        detections[0, 2, 22:] += np.arange(34) * np.random.random() * 0.01

        nptest.assert_almost_equal(pose_nms(detections), np.array(
            [[[1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
               1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
               1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
               1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
               1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
               0.00000000e+00, 0.00000000e+00, 5.05930847e-03, 1.00505931e+00,
               2.00505931e+00, 3.00505931e+00, 4.00505931e+00, 5.00505931e+00,
               6.00505931e+00, 7.00505931e+00, 8.00505931e+00, 9.00505931e+00,
               1.00050593e+01, 1.10050593e+01, 1.20050593e+01, 1.30050593e+01,
               1.40050593e+01, 1.50050593e+01, 1.60050593e+01, 1.70050593e+01,
               1.80050593e+01, 1.90050593e+01, 2.00050593e+01, 2.10050593e+01,
               2.20050593e+01, 2.30050593e+01, 2.40050593e+01, 2.50050593e+01,
               2.60050593e+01, 2.70050593e+01, 2.80050593e+01, 2.90050593e+01,
               3.00050593e+01, 3.10050593e+01, 3.20050593e+01, 3.30050593e+01]]]
        ), decimal=1)
