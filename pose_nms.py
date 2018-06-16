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

def get_keypoint_width(pose_loc):
    """
    :param pose_loc: [n x 17 x 2]
    :return:
    """
    alpha = 0.1
    body_width = max(
        np.max(pose_loc[:, :, 0]) - np.min(pose_loc[:, :, 0]),
        np.max(pose_loc[:, :, 1]) - np.min(pose_loc[:, :, 1]))
    keypoint_width = body_width * alpha
    return keypoint_width

def get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width,
                 delta1, delta2, mu):
    """
    <class 'numpy'>
    :param choose_idx:
    :param pose_conf: [n x 17]
    :param box_conf: [n x 1]
    :param pose_loc: [n x 17 x 2]
    :param box_loc: [n x 4]
    :return:
    """
    num_pose = pose_conf.shape[0]
    target_pose_conf = pose_conf[choose_idx]

    dist = np.sqrt(np.sum(np.square(pose_loc - pose_loc[choose_idx]), axis=-1)) / keypoint_width
    mask = (dist <= 1)

    score_dists = np.zeros([num_pose, 17])
    pose_conf_tile = np.tile(target_pose_conf, [pose_conf.shape[0], 1])
    score_dists[mask] = np.tanh(pose_conf_tile[mask] / delta1) * np.tanh(pose_conf[mask] / delta1)

    point_dists = np.exp((-1) * dist / delta2)
    return np.sum(score_dists, axis=1) + mu * np.sum(point_dists, axis=1)


def PCK_match(choose_idx, pose_loc, keypoint_width):
    """
    :param choose_idx:
    :param pose_loc: [n x 17 x 2]
    :param keypoint_width: 0.1 x body_width
    :return:
    """
    dist = np.sqrt(np.sum(np.square(pose_loc - pose_loc[choose_idx]), axis=-1)) / keypoint_width
    num_match_keypoints = np.sum(dist / min(keypoint_width, 7) <= 1, axis=1)
    face_index = np.zeros(dist.shape)
    face_index[:, :5] = 1

    face_match_keypoints = np.sum((dist / 10 <= 1) & (face_index == 1), axis=1)
    return num_match_keypoints, face_match_keypoints


if __name__ == "__main__":
    detections = np.load("./detections.save.npz")
    detections = detections["arr_0"]
    detections = np.array(detections)
    pose_conf = detections[0, :, 1:18]
    box_conf = detections[0, :, 0]
    pose_loc = detections[0, :, 22:]
    box_loc = detections[0, :, 18:22]

    candidates = np.arange(detections.shape[1])
    pose_loc = np.reshape(pose_loc, [pose_loc.shape[0], -1, 2])

    while candidates.size > 0:
        choose_idx = np.argmax(box_conf[candidates])
        choose = candidates[choose_idx]

        keypoint_width = get_keypoint_width(pose_loc)
        simi = get_pose_dis(choose_idx, pose_conf[candidates],
                      pose_loc[candidates],keypoint_width=keypoint_width,
                            delta1=1, delta2=1, mu=1)
        num_match_keypoints, _ = PCK_match(choose_idx, pose_loc[candidates],
                                           keypoint_width)


        # candidates = np.delete(candidates, choose_idx)



    print(candidates)












