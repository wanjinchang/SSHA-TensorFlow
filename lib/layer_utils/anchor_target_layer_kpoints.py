#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: anchor_target_layer_kpoints.py
@time: 18-6-23 上午9:38
@desc: modify from https://github.com/rbgirshick/py-faster-rcnn
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.model.config import cfg
import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps
from lib.model.bbox_transform import bbox_transform, kpoints_transform


def anchor_target_layer(rpn_cls_prob, gt_boxes, gt_points, im_info, _feat_stride, all_anchors, num_anchors, target_name):
    # def anchor_target_layer(anchor_hw, rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors
    hard_mining = cfg.TRAIN.HARD_POSITIVE_MINING

    # allow boxes to sit over the edge by a small amount
    # _allowed_border = 0
    # follow the SSH setting
    if target_name == "M3":
        _allowed_border = 512
    else:
        _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_prob.shape[1:3]

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # only keep anchors inside anchors
    # keep away the problem of ‘ValueError: attempt to get argmax of an empty sequence’ during training
    if inds_inside.shape[0] == 0:
        # If no anchors inside use whatever anchors we have
        inds_inside = np.arange(0, total_anchors)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    if cfg.TRAIN.FORCE_FG_FOR_EACH_GT:
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    ################################### Subsample positive labels ##################################
    # subsample positive labels if we have too many
    # num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    # fg_inds = np.where(labels == 1)[0]
    # if len(fg_inds) > num_fg:
    #     disable_inds = npr.choice(
    #         fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    #     labels[disable_inds] = -1

    ##################### Add OHEM for subsample positive labels(Online Hard Examples Mining) ##########
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        if hard_mining:
            ohem_scores = rpn_cls_prob[:, :, :, num_anchors:]
            ohem_scores = ohem_scores.reshape((-1, 1))
            ohem_scores = ohem_scores[inds_inside]
            pos_ohem_scores = 1 - ohem_scores[fg_inds]
            order_pos_ohem_scores = pos_ohem_scores.ravel().argsort()[::-1]
            ohem_sampled_fgs = fg_inds[order_pos_ohem_scores[:num_fg]]
            labels[fg_inds] = -1
            labels[ohem_sampled_fgs] = 1
        else:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
    ########################################## End ##################################################


    ##########################################Subsample negative labels #############################
    # subsample negative labels if we have too many
    # num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    # bg_inds = np.where(labels == 0)[0]
    # if len(bg_inds) > num_bg:
    #     disable_inds = npr.choice(
    #         bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    #     labels[disable_inds] = -1

    ################# Add OHEM for subsample negative labels(Online Hard Examples Mining) ############
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        if not hard_mining:
            # randomly sub-sampling negatives
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
        else:
            # sort ohem scores
            ohem_scores = rpn_cls_prob[:, :, :, num_anchors:]
            ohem_scores = ohem_scores.reshape((-1, 1))
            ohem_scores = ohem_scores[inds_inside]
            neg_ohem_scores = ohem_scores[bg_inds]
            order_neg_ohem_scores = neg_ohem_scores.ravel().argsort()[::-1]
            ohem_sampled_bgs = bg_inds[order_neg_ohem_scores[:num_bg]]
            labels[bg_inds] = -1
            labels[ohem_sampled_bgs] = 0
    ########################################## End ##############################################

    # Compute boxes regression targets
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_bboxes_targets(anchors, gt_boxes[argmax_overlaps, :])

    # Compute kpoints offset targets
    kpoints_targets = np.zeros((len(inds_inside), 10), dtype=np.float32)
    kpoints_targets = _compute_kpoints_targets(anchors, gt_points[argmax_overlaps, :10])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    # only the positive ones have regression targets
    kpoints_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
    kpoints_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_KPOINTS_POSITIVE_WEIGHTS)


    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        kpoints_positive_weights = np.ones((1, 10)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
        kpoints_positive_weights = (cfg.TRAIN.RPN_KPOINTS_POSITIVE_WEIGHTS /
                            np.sum(labels == 1))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    kpoints_weights[labels == 1, :] = kpoints_positive_weights
    if gt_points.size > 0:
        gt_points_flag = gt_points[argmax_overlaps, 10]
        gt_points_flag = np.array(gt_points_flag, np.int32)
        kpoints_weights[gt_points_flag == 0, :] = np.array(cfg.TRAIN.RPN_KPOINTS_WEIGHTS_NON)

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    kpoints_targets = _unmap(kpoints_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    kpoints_weights = _unmap(kpoints_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    # kpoints_targets
    kpoints_targets = kpoints_targets.reshape((1, height, width, A * 10))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights

    kpoints_weights = kpoints_weights.reshape((1, height, width, A * 10))
    rpn_kpoints_weights = kpoints_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, kpoints_targets, \
           rpn_kpoints_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_bboxes_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _compute_kpoints_targets(anchors, gt_kpoints):
    """Compute landmarks offset targets for an image."""
    return kpoints_transform(anchors, gt_kpoints).astype(np.float32, copy=False)


