# --------------------------------------------------------
# SSH
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def kpoints_transform(anchors, gt_kpoints):
    """
    Transform the set of class-agnostic kpoints into class-specific points
    by applying the predicted offsets (kpoints_deltas)
    :param anchors: !important [N 4]
    :param gt_kpoints: [N, 10]
    :return: [N, 10]
    """
    anchors = anchors.astype(np.float, copy=False)
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    target_p0 = (gt_kpoints[:, 0] - ctr_x) / (widths + 1e-14)
    target_p1 = (gt_kpoints[:, 1] - ctr_y) / (heights + 1e-14)
    target_p2 = (gt_kpoints[:, 2] - ctr_x) / (widths + 1e-14)
    target_p3 = (gt_kpoints[:, 3] - ctr_y) / (heights + 1e-14)
    target_p4 = (gt_kpoints[:, 4] - ctr_x) / (widths + 1e-14)
    target_p5 = (gt_kpoints[:, 5] - ctr_y) / (heights + 1e-14)
    target_p6 = (gt_kpoints[:, 6] - ctr_x) / (widths + 1e-14)
    target_p7 = (gt_kpoints[:, 7] - ctr_y) / (heights + 1e-14)
    target_p8 = (gt_kpoints[:, 8] - ctr_x) / (widths + 1e-14)
    target_p9 = (gt_kpoints[:, 9] - ctr_y) / (heights + 1e-14)
    kpoints_targets = np.vstack(
        (target_p0, target_p1, target_p2, target_p3, target_p4, target_p5, target_p6, target_p7, target_p8, target_p9)).transpose()

    return kpoints_targets

def clip_kpoints(points, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """

    points[:, 0::10] = np.maximum(np.minimum(points[:, 0::10], im_shape[1] - 1), 0)
    points[:, 1::10] = np.maximum(np.minimum(points[:, 1::10], im_shape[0] - 1), 0)
    points[:, 2::10] = np.maximum(np.minimum(points[:, 2::10], im_shape[1] - 1), 0)
    points[:, 3::10] = np.maximum(np.minimum(points[:, 3::10], im_shape[0] - 1), 0)
    points[:, 4::10] = np.maximum(np.minimum(points[:, 4::10], im_shape[1] - 1), 0)
    points[:, 5::10] = np.maximum(np.minimum(points[:, 5::10], im_shape[0] - 1), 0)
    points[:, 6::10] = np.maximum(np.minimum(points[:, 6::10], im_shape[1] - 1), 0)
    points[:, 7::10] = np.maximum(np.minimum(points[:, 7::10], im_shape[0] - 1), 0)
    points[:, 8::10] = np.maximum(np.minimum(points[:, 8::10], im_shape[1] - 1), 0)
    points[:, 9::10] = np.maximum(np.minimum(points[:, 9::10], im_shape[0] - 1), 0)

    return points

def kpoints_transform_inv(anchors, point_deltas):
    """
    Transform the set of class-agnostic landmarks into class-specific points
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if anchors.shape[0] == 0:
        return np.zeros((0, point_deltas.shape[1]))

    anchors = anchors.astype(np.float, copy=False)
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    d1x = point_deltas[:, 0]
    d1y = point_deltas[:, 1]
    d2x = point_deltas[:, 2]
    d2y = point_deltas[:, 3]
    d3x = point_deltas[:, 4]
    d3y = point_deltas[:, 5]
    d4x = point_deltas[:, 6]
    d4y = point_deltas[:, 7]
    d5x = point_deltas[:, 8]
    d5y = point_deltas[:, 9]


    pred_points = np.zeros(point_deltas.shape)
    # x1
    x = d1x * widths
    print("aa", d1x.shape, widths.shape, ctr_x.shape, x.shape)
    pred_points[:, 0] = d1x * widths + ctr_x
    pred_points[:, 1] = d1y * heights + ctr_y
    pred_points[:, 2] = d2x * widths + ctr_x
    pred_points[:, 3] = d2y * heights + ctr_y
    pred_points[:, 4] = d3x * widths + ctr_x
    pred_points[:, 5] = d3y * heights + ctr_y
    pred_points[:, 6] = d4x * widths + ctr_x
    pred_points[:, 7] = d4y * heights + ctr_y
    pred_points[:, 8] = d5x * widths + ctr_x
    pred_points[:, 9] = d5y * heights + ctr_y

    return pred_points

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    ctr_x = tf.add(boxes[:, 0], widths * 0.5)
    ctr_y = tf.add(boxes[:, 1], heights * 0.5)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)

def kpoints_transform_inv_tf(anchors, points_deltas):
    anchors = tf.cast(anchors, points_deltas.dtype)
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    d1x = points_deltas[:, 0]
    d1y = points_deltas[:, 1]
    d2x = points_deltas[:, 2]
    d2y = points_deltas[:, 3]
    d3x = points_deltas[:, 4]
    d3y = points_deltas[:, 5]
    d4x = points_deltas[:, 6]
    d4y = points_deltas[:, 7]
    d5x = points_deltas[:, 8]
    d5y = points_deltas[:, 9]

    p0 = d1x * widths + ctr_x
    p1 = d1y * heights + ctr_y
    p2 = d2x * widths + ctr_x
    p3 = d2y * heights + ctr_y
    p4 = d3x * widths + ctr_x
    p5 = d3y * heights + ctr_y
    p6 = d4x * widths + ctr_x
    p7 = d4y * heights + ctr_y
    p8 = d5x * widths + ctr_x
    p9 = d5y * heights + ctr_y
    return tf.stack([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=1)

def clip_boxes_tf(boxes, im_info):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)

def clip_kpoints_tf(kpoints, im_info):
    p0 = tf.maximum(tf.minimum(kpoints[:, 0], im_info[1] - 1), 0)
    p1 = tf.maximum(tf.minimum(kpoints[:, 1], im_info[0] - 1), 0)
    p2 = tf.maximum(tf.minimum(kpoints[:, 2], im_info[1] - 1), 0)
    p3 = tf.maximum(tf.minimum(kpoints[:, 3], im_info[0] - 1), 0)
    p4 = tf.maximum(tf.minimum(kpoints[:, 4], im_info[1] - 1), 0)
    p5 = tf.maximum(tf.minimum(kpoints[:, 5], im_info[0] - 1), 0)
    p6 = tf.maximum(tf.minimum(kpoints[:, 6], im_info[1] - 1), 0)
    p7 = tf.maximum(tf.minimum(kpoints[:, 7], im_info[0] - 1), 0)
    p8 = tf.maximum(tf.minimum(kpoints[:, 8], im_info[1] - 1), 0)
    p9 = tf.maximum(tf.minimum(kpoints[:, 9], im_info[0] - 1), 0)
    return tf.stack([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=1)
