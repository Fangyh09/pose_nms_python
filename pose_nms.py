# -*- coding: utf-8 -*-
"""
@author: fangyh09
"""

import numpy as np

TEST_MODE = True

# delta1 = 1;
# mu = 1.7;
# delta2 = 2.65;
# gamma = 22.48;
def pose_nms(detections, mu=1.7, delta1=1, delta2=2.65, gamma=22.48):
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
    # pose_conf: [n x 17]
    # box_conf: [n x 1]
    # pose_loc: [n x 17 x 2]
    # box_loc: [n x 4]
    pose_conf = detections[0, :, 1:18]
    box_conf = detections[0, :, 0]
    pose_loc = detections[0, :, 22:]
    box_loc = detections[0, :, 18:22]

    candidates = np.arange(detections[0, :, :].shape[0])
    nums_candidates = len(candidates)
    # todo(delete it)
    # candidates = np.array([0, 8, 10])
    pose_loc = np.reshape(pose_loc, [pose_loc.shape[0], -1, 2])

    matchThreds = 5
    scoreThreds = 0.3

    merge_ids = {}
    choose_set = []
    while candidates.size > 0:
        choose_idx = np.argmax(box_conf[candidates])
        choose = candidates[choose_idx]

        keypoint_width = get_keypoint_width(pose_loc[choose])
        simi = get_pose_dis(choose_idx, pose_conf[candidates],
                            pose_loc[candidates], keypoint_width=keypoint_width,
                            delta1=delta1, delta2=delta2, mu=mu)
        num_match_keypoints, _ = PCK_match(choose_idx, pose_loc[candidates],
                                           keypoint_width)

        delete_ids = np.arange(candidates.shape[0])[
            (simi > gamma) | (num_match_keypoints >= matchThreds)]

        assert (delete_ids.size > 0)  # at least itself
        if delete_ids.size == 0:
            delete_ids = choose_idx

        merge_ids[choose] = candidates[delete_ids]
        choose_set.append(choose)

        # force to delete itself
        candidates = np.delete(candidates, np.append(delete_ids, choose_idx))

    print(choose_set)

    if TEST_MODE:
        assert (np.sum([merge_ids[key].shape[0] for key in choose_set])
                == nums_candidates)

    # merge poses

    result_detections = []
    for root_pose_idx in choose_set:
        simi_poses_idx = merge_ids[root_pose_idx]
        max_score = np.max(pose_conf[simi_poses_idx, :])
        if max_score < scoreThreds:
            continue
        keypoint_width = get_keypoint_width(pose_loc[root_pose_idx])

        merge_poses, merge_score = merge_pose(pose_loc[root_pose_idx],
                                              pose_loc[simi_poses_idx],
                                              pose_conf[simi_poses_idx],
                                              keypoint_width)
        max_score = np.max(merge_poses)
        if max_score < scoreThreds:
            continue
        print(merge_poses, merge_score)

        merge_detection = np.zeros([1, 56])
        # pose_conf = detections[0, :, 1:18]
        # box_conf = detections[0, :, 0]
        # pose_loc = detections[0, :, 22:]
        # box_loc = detections[0, :, 18:22]
        merge_detection[0, 1:18] = merge_score
        merge_detection[0, 22:] = merge_poses.reshape(-1)
        # use root bbox
        merge_detection[0, 0] = box_conf[root_pose_idx]
        merge_detection[0, 18:22] = box_loc[root_pose_idx]

        # result_detections = np.append(result_detections, merge_detection,
        #                               axis=0)
        result_detections.append(merge_detection)
    # result_detections = np.expand_dims(result_detections, axis=0)
    result_detections = np.array(result_detections)
    return result_detections


def get_keypoint_width(pose_loc):
    """
    :param pose_loc: [17 x 2]
    :return:
    """
    alpha = 0.1
    body_width = max(
        np.max(pose_loc[:, 0]) - np.min(pose_loc[:, 0]),
        np.max(pose_loc[:, 1]) - np.min(pose_loc[:, 1]))
    keypoint_width = body_width * alpha
    return keypoint_width


def get_pose_dis(choose_idx, pose_conf, pose_loc, keypoint_width,
                 delta1, delta2, mu):
    """
    <class 'numpy'>
    :param choose_idx:
    :param pose_conf: [n x 17]
    :param pose_loc: [n x 17 x 2]
    :return:
    """
    # make sure dim
    if pose_conf.ndim == 1:
        pose_conf = np.expand_dims(pose_conf, axis=0)
        print("make sure pose_conf dim")
    if pose_loc.ndim == 2:
        pose_loc = np.expand_dims(pose_loc, axis=0)
        print("make sure pose_loc dim")

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
    # make sure dim
    if pose_loc.ndim == 2:
        pose_loc = np.expand_dims(pose_loc, axis=0)
        print("make sure pose_loc dim")
    dist = np.sqrt(np.sum(np.square(pose_loc - pose_loc[choose_idx]), axis=-1))
    num_match_keypoints = np.sum(dist / min(keypoint_width, 7) <= 1, axis=1)
    face_index = np.zeros(dist.shape)
    face_index[:, :5] = 1

    face_match_keypoints = np.sum((dist / 10 <= 1) & (face_index == 1), axis=1)
    return num_match_keypoints, face_match_keypoints


def merge_pose(root_pose, cluster_pose_loc, cluster_pose_conf, keypoint_width):
    """
    root_pose must be in cluster_pose_loc.
    :param root_pose: [17 x 2]
    :param cluster_pose_loc: [n x 17 x 2]
    :param cluster_pose_conf: [n x 17]
    :param keypoint_width: 0.1 x body_width
    :return:
    """
    # make sure dim
    if cluster_pose_loc.ndim == 2:
        cluster_pose_loc = np.expand_dims(cluster_pose_loc, axis=0)
        print("make sure cluster_pose_loc dim")
    if cluster_pose_conf.ndim == 1:
        cluster_pose_conf = np.expand_dims(cluster_pose_conf, axis=0)
        print("make sure cluster_pose_conf dim")
    dist = np.sqrt(np.sum(np.square(cluster_pose_loc - root_pose), axis=-1)) / keypoint_width
    keypoint_width = min(keypoint_width, 15)
    mask = (dist <= keypoint_width)
    # final_pose = np.zeros([17,2]); final_scores = np.zeros(17)

    pose_loc_t = cluster_pose_loc * np.expand_dims(mask, axis=-1)
    pose_conf_t = cluster_pose_conf * mask

    weighted_pose_conf = pose_conf_t * 1.0 / np.sum(pose_conf_t, axis=0)
    final_scores = np.sum(weighted_pose_conf, axis=0)

    weighted_pose_loc = pose_loc_t[:, :, :] * np.expand_dims(weighted_pose_conf, -1)
    final_pose = np.sum(weighted_pose_loc, axis=0)

    return final_pose, final_scores


if __name__ == "__main__":
    detections = np.load("./detections.save.npz")
    detections = detections["arr_0"]
    detections = np.array(detections)

    pose_nms(detections)



