# -*- coding: utf-8 -*-
"""
@author: fangyh09
"""

import numpy as np


def pose_nms(detections):
    """
    :param detections: shape: [1xnx(1+17+4+34)] <class 'numpy'>
    Example:
    ( 0 ,.,.) =
      0.9196  0.8745  0.1870  ...   0.1308  0.0363  0.1306
      0.9043  0.6790  0.4925  ...   0.1503  0.1631  0.3767
      0.8862  0.7278  0.4409  ...   0.4891  0.0785  0.5004
               ...             ⋱             ...
      0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
      0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
      0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
    [size 1x100x56]
    :return:
    """
    pass


def get_pose_dis(choose_idx, pose_conf, box_conf, pose_loc, box_loc,
                 delta1, delta2, mu):
    """
    <class 'numpy'>
    :param candidates: array of idxs
    :param pose_conf: [n x 17]
    :param box_conf: [n x 1]
    :param pose_loc: [n x 34]
    :param box_loc: [n x 4]
    :return:
    """
    dist = np.square(pose_loc - pose_loc[choose_idx])
    pass


if __name__ == "__main__":
    detections = np.load("./detections.save.npz")
    detections = detections["arr_0"]
    detections = np.array(detections)
    pose_conf = detections[0, :, 1:18]
    box_conf = detections[0, :, 0]
    pose_loc = detections[0, :, 22:]
    box_loc = detections[0, :, 18:22]

    candidates = np.arange(detections.shape[1])
    pose_loc = np.reshape(pose_loc, [pose_loc.shape[0], pose_loc.shape[1], -1, 2])

    alpha = 0.1
    while candidates.size > 0:
        choose_idx = np.argmax(box_conf[candidates])
        choose = candidates[choose_idx]

        body_width = np.max(
            np.max(pose_loc[candidates,:,0]) - np.min(pose_loc[candidates,:,0]),
            np.max(pose_loc[candidates,:,1]) - np.min(pose_loc[candidates,:,1]))
        keypoint_width = body_width * alpha

        dist = np.sqrt(np.sum(np.square(pose_loc - pose_loc[choose_idx]), axis=-1)) / keypoint_width



        # candidates = np.delete(candidates, choose_idx)



    print(candidates)












