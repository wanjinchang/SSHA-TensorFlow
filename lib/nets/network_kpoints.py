#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: network_kpoints.py
@time: 18-6-22 上午9:38
@desc: modify from Xinlei Chen
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from lib.layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from lib.layer_utils.proposal_layer_kpoints import proposal_layer, proposal_layer_tf
from lib.layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
# from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.anchor_target_layer_kpoints import anchor_target_layer
# from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.utils.visualization import draw_bounding_boxes
from lib.utils.timer import Timer

from lib.model.config import cfg


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._all_preds = {}
        self._losses = {}
        self._all_losses = {}
        self._anchors = {}
        self._anchor_targets = {}
        self._all_anchor_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image, self._gt_boxes, self._im_info],
                           tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._mode,
                    self._feat_stride[name],
                    self._anchors,
                    self._num_anchors[name]
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                               self._feat_stride[name], self._anchors, self._num_anchors[name]],
                                              [tf.float32, tf.float32], name="proposal_top")

            tf.reshape(rois, [-1, 5])
            tf.reshape(rpn_scores, [-1, 1])
        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, kpoints_pred, target_name, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores, points = proposal_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    kpoints_pred,
                    self._im_info,
                    self._mode,
                    self._feat_stride[target_name],
                    self._anchors,
                    self._num_anchors[target_name]
                )
            else:
                rois, rpn_scores, points = tf.py_func(proposal_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, kpoints_pred, self._im_info, self._mode,
                                               self._feat_stride[target_name], self._anchors, self._num_anchors[target_name]],
                                              [tf.float32, tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])
            points.set_shape([None, 10])

        return rois, rpn_scores, points

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_prob, scope_name, target_name):
        with tf.variable_scope(scope_name + target_name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, \
            kpoints_targets, rpn_kpoints_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_prob, self._gt_boxes, self.gt_kpoints, self._im_info, self._feat_stride[target_name],
                 self._anchors, self._num_anchors[target_name], target_name],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors[target_name] * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors[target_name] * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors[target_name] * 4])
            kpoints_targets.set_shape([1, None, None, self._num_anchors[target_name] * 10])
            rpn_kpoints_weights.set_shape([1, None, None, self._num_anchors[target_name] * 10])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")

            rpn_params = {}
            rpn_params['rpn_labels'] = rpn_labels
            rpn_params['rpn_bbox_targets'] = rpn_bbox_targets
            rpn_params['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            rpn_params['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
            rpn_params['kpoints_targets'] = kpoints_targets
            rpn_params['rpn_kpoints_weights'] = rpn_kpoints_weights
            self._anchor_targets[target_name] = rpn_params

        return rpn_labels

    def _anchor_component(self, rpn_cls_score, name):
        with tf.variable_scope('ANCHOR_' + name + '_' + self._tag) as scope:
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(
                    rpn_cls_score,
                    self._feat_stride[name],
                    self._anchor_scales[name],
                    self._anchor_ratios
                )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [rpn_cls_score,
                                                     self._feat_stride[name], self._anchor_scales[name],
                                                     self._anchor_ratios],
                                                    [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # self._image_to_head --> vgg conv5_3
        timer = Timer()
        timer.tic()
        self._image_to_head(is_training)
        timer.toc()
        print('base_network took {:.3f}s'.format(timer.total_time))
        with tf.variable_scope(self._scope, self._scope):
            with arg_scope([slim.conv2d], weights_initializer=initializer):
                self._ssh_region_proposal(is_training)

        if 'M1' in self._feat_branches:
            self.all_rois_scores = tf.concat([self._predictions["M1"]["rois_scores"], self._predictions["M2"]["rois_scores"],
                                         self._predictions["M3"]["rois_scores"]], axis=0, name='roi_scores')
            self.all_rois = tf.concat([self._predictions["M1"]["rois"], self._predictions["M2"]["rois"], self._predictions["M3"]["rois"]],
                                  axis=0, name='rois')
            self.all_kpoints = tf.concat([self._predictions["M1"]["kpoints"], self._predictions["M2"]["kpoints"],
                                          self._predictions["M3"]["kpoints"]],
                                         axis=0, name='kpoints')
        else:
            self.all_rois_scores = tf.concat([self._predictions["M2"]["rois_scores"], self._predictions["M3"]["rois_scores"]], axis=0, name='roi_scores')
            self.all_rois = tf.concat([self._predictions["M2"]["rois"], self._predictions["M3"]["rois"]], axis=0, name='rois')
            self.all_kpoints = tf.concat([self._predictions["M2"]["kpoints"], self._predictions["M3"]["kpoints"]],
                                         axis=0, name='kpoints')

        self._all_preds['all_predictions'] = self._predictions
        self._all_anchor_targets['all_anchors'] = self._anchor_targets
        self._score_summaries.update(self._all_preds)
        self._score_summaries.update(self._all_anchor_targets)

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _localization_loss(self, kpoints_pred, kpoints_targets, weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        kpoints_diff = kpoints_pred - kpoints_targets
        abs_in_kpoints_diff = tf.abs(kpoints_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_kpoints_diff, 1. / sigma_2)))
        in_loss_kpoints = tf.pow(kpoints_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_kpoints_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = weights * in_loss_kpoints
        loss_kpoints = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_kpoints

    def _compute_losses(self):
        self._losses['total_loss'] = 0
        for k in self._feat_branches:
            loss = self._add_losses(k)
            self._losses['total_loss'] += loss
        total_loss = self._losses['total_loss']
        self._all_losses['all_losses'] = self._losses
        self._event_summaries.update(self._all_losses)
        return total_loss

    def _add_losses(self, name, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + name + '_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions[name]['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets[name]['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions[name]['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets[name]['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets[name]['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets[name]['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RPN, landmarks loss
            kpoints_pred = self._predictions[name]['kpoints_pred']
            kpoints_targets = self._anchor_targets[name]['kpoints_targets']

            rpn_kpoints_weights = self._anchor_targets[name]['rpn_kpoints_weights']
            rpn_loss_kpoints = self._localization_loss(kpoints_pred, kpoints_targets, rpn_kpoints_weights, sigma=sigma_rpn,  dim=[1,2,3])

            loss_items = {}
            loss_items['rpn_cross_entropy'] = rpn_cross_entropy
            loss_items['rpn_loss_box'] = rpn_loss_box
            loss_items['kpoints_loss'] = rpn_loss_kpoints

            loss = rpn_cross_entropy + rpn_loss_box + rpn_loss_kpoints
            # loss = rpn_cross_entropy + rpn_loss_box
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            loss_regu = loss + regularization_loss
            loss_items['total_loss'] = loss_regu
            self._losses[name] = loss_items

        return loss_regu

    def context_module(self, inputs, out_channels):
        net = inputs
        with tf.variable_scope("context_module"):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                context_conv_1 = slim.conv2d(net, out_channels // 2, [3, 3], scope='conv1')
                context_conv_2 = slim.conv2d(context_conv_1, out_channels // 2, [3, 3], scope='conv2')
                context_conv_3 = slim.conv2d(context_conv_1, out_channels // 2, [3, 3], scope='conv3')
                context_conv_3 = slim.conv2d(context_conv_3, out_channels // 2, [3, 3], scope='conv4')
                context_model = tf.concat([context_conv_2, context_conv_3], axis=3, name='concat')
        return context_model

    def DetectionModule(self, inputs, out_channels):
        net = inputs
        detection_conv_model = slim.conv2d(net, out_channels, [3, 3], scope="conv1")
        context_model = self.context_module(net, out_channels)
        detection_model = tf.concat([detection_conv_model, context_model], axis=3, name='concat')
        return detection_model

    def _ssh_region_proposal(self, is_training):
        """
        Modify the original region proposal network from faster rcnn to ssh architecture.
        """
        end_points = {}

        # the Module M3 and M2 of ssh is based on vgg conv5_3 features
        if 'M3' in self._feat_branches:
            print('contained large faces branch M3!!!!!!')
            end_point = 'M3'
            with tf.variable_scope(end_point):
                feat_layer = self._feat_layers[end_point]
                if self._scope == 'MobilenetV2':
                    net = self.end_points[feat_layer]
                else:
                    net = slim.max_pool2d(self.end_points[feat_layer], [2, 2], scope='pool1', padding='VALID')
                net = self.DetectionModule(net, self._Module_boxes[end_point])
                end_points[end_point] = net

        if 'M2' in self._feat_branches:
            end_point = "M2"
            with tf.variable_scope(end_point):
                feat_layer = self._feat_layers[end_point]
                net = self.DetectionModule(self.end_points[feat_layer], self._Module_boxes[end_point])
                # print("M2:", net.shape)
                end_points[end_point] = net

        # the Module M1 of ssh is based on vgg conv4_3 features
        if 'M1' in self._feat_branches:
            end_point = "M1"
            with tf.variable_scope(end_point):
                feat_layers = self._feat_layers[end_point]
                M1_dimReduction_1 = slim.conv2d(self.end_points[feat_layers[0]], 128, [1, 1], scope="conv1_1", padding='VALID')
                M1_dimReduction_2 = slim.conv2d(self.end_points[feat_layers[1]], 128, [1, 1], scope="conv1_2", padding='VALID')
                # print("######:", M1_dimReduction_2.shape)   -->(1, ?, ?, 128)
                M1_dimReduction_2 = tf.image.resize_bilinear(M1_dimReduction_2, tf.shape(M1_dimReduction_1)[1:3])
                M1_elementWiseSum = tf.add(M1_dimReduction_1, M1_dimReduction_2)
                net = slim.conv2d(M1_elementWiseSum, 128, [3, 3], scope="conv2")
                net = self.DetectionModule(net, self._Module_boxes[end_point])
                # net = self.DetectionModule(net, 128)
                end_points[end_point] = net

        # for k, v in self._Module_boxes.items():
        for k in self._feat_branches:
            with tf.variable_scope(k + "_box"):
                cls_score, cls_score_reshape, cls_prob, cls_pred, bbox_pred, kpoints_pred, rois, rois_scores, kpoints = self._region_proposal(end_points[k], k,
                                                                                                    is_training)
                pred = {}
                pred["rpn_cls_score"] = cls_score
                pred["rpn_cls_score_reshape"] = cls_score_reshape
                pred["rpn_cls_prob"] = cls_prob
                pred["rpn_cls_pred"] = cls_pred
                pred["rpn_bbox_pred"] = bbox_pred
                pred["kpoints_pred"] = kpoints_pred
                pred["rois"] = rois
                pred["rois_scores"] = rois_scores
                pred["kpoints"] = kpoints
                self._predictions[k] = pred
        return end_points

    def _region_proposal(self, net_conv, name, is_training):
        rpn_cls_score = slim.conv2d(net_conv, self._num_anchors[name] * 2, [1, 1], trainable=is_training,
                                    padding='VALID', activation_fn=None, scope=name + '_rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, name + '_rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, name + "_rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name=name + "_rpn_cls_pred")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors[name] * 2, name + "_rpn_cls_prob")

        # bbox pred
        rpn_bbox_pred = slim.conv2d(net_conv, self._num_anchors[name] * 4, [1, 1], trainable=is_training,
                                    padding='VALID', activation_fn=None, scope=name + '_rpn_bbox_pred')

        ## kpoints pred
        kpoints_pred = slim.conv2d(net_conv, self._num_anchors[name] * 10, [1, 1], trainable=is_training,
                                     padding='VALID', activation_fn=None, scope=name + '_rpn_kpoints_pred')

        self._anchor_component(rpn_cls_score, name)
        if is_training:
            rois, roi_scores, kpoints = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, kpoints_pred, name, name + "_rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_prob, "anchor_", name)
        else:
            if cfg.TEST.MODE == 'nms':
                print('test mode is: nms')
                rois, roi_scores, kpoints = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, kpoints_pred, name, name + "_rois")
            elif cfg.TEST.MODE == 'top':
                print('test mode is: top')
                rois, roi_scores = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, name)
            else:
                raise NotImplementedError
        return rpn_cls_score, rpn_cls_score_reshape, rpn_cls_prob, rpn_cls_pred, rpn_bbox_pred, kpoints_pred, rois, roi_scores, kpoints

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_scales=(8, 16, 32), anchor_ratios=(1,)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image_tensor')
        self._im_info = tf.placeholder(tf.float32, shape=[3], name='im_info')
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self._tag = tag
        self.gt_kpoints = tf.placeholder(tf.float32, shape=[None, 11], name='gt_kpoints')
        # self.landmarks_flag = tf.constant(landmarks_flag, dtype=tf.int32)
        self._num_classes = num_classes
        self._mode = mode

        # get the anchor numbers corresponding to the three branches
        self._anchor_scales = anchor_scales
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(self._anchor_ratios)
        self._num_anchors = {}
        for branch in self._feat_branches:
            num_scales = len(self._anchor_scales[branch])
            num_anchors = num_scales * self._num_ratios
            self._num_anchors[branch] = num_anchors

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding="SAME",
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self._build_network(training)

        # layers_to_output = {'rois': rois}
        layers_to_output = {}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            for k in self._feat_branches:
                stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_anchors[k]))
                means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_anchors[k]))
                self._predictions[k]["rpn_bbox_pred"] *= stds
                self._predictions[k]["rpn_bbox_pred"] += means
        else:
            # self._add_losses()
            self._compute_losses()
            layers_to_output.update(self._all_losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    for k1, v1 in var.items():
                        if isinstance(v1, dict):
                            for k2, v2 in v1.items():
                                val_summaries.append(tf.summary.scalar(key + '_' + k1 + '_' + k2, v2))
                        else:
                            val_summaries.append(tf.summary.scalar(key + '_' + k1, v1))
                for key, var in self._score_summaries.items():
                    for k1, v1 in var.items():
                        if isinstance(v1, dict):
                            for k2, v2 in v1.items():
                                self._add_score_summary(key + '_' + k1 + '_' + k2, v2)
                        else:
                            self._add_score_summary(key + '_' + k1, v1)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image, self._im_info: im_info}
        # cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
        #                                                  self._predictions['cls_prob'],
        #                                                  self._predictions['bbox_pred'],
        #                                                  self._predictions['rois']],
        #                                                 feed_dict=feed_dict)
        timer = Timer()
        timer.tic()
        predictions = sess.run(self._predictions, feed_dict=feed_dict)
        timer.toc()
        print('Prediction took {:.3f}s'.format(timer.total_time))

        # keep M1 M2 M3 branch to detect small/medium/large faces
        if 'M1' in self._feat_branches:
            cls_prob = np.concatenate((predictions["M1"]["rois_scores"], predictions["M2"]["rois_scores"],
                                       predictions["M3"]["rois_scores"]), axis=0)
            rois = np.concatenate((predictions["M1"]["rois"], predictions["M2"]["rois"], predictions["M3"]["rois"]), axis=0)
            kpoints = np.concatenate((predictions["M1"]["kpoints"], predictions["M2"]["kpoints"], predictions["M3"]["kpoints"]), axis=0)


        # discard M1 branch, only keep M2 and M3 branches to detect medium and large faces
        else:
            print('do not contain M1 branch!!!!')
            cls_prob = np.concatenate((predictions["M2"]["rois_scores"], predictions["M3"]["rois_scores"]), axis=0)
            rois = np.concatenate((predictions["M2"]["rois"], predictions["M3"]["rois"]), axis=0)
            kpoints = np.concatenate((predictions["M2"]["kpoints"], predictions["M3"]["kpoints"]), axis=0)
        return cls_prob, rois, kpoints

    def test_image_without_gen_anchors(self, sess, image, im_info):
        feed_dict = {self._image: image, self._im_info: im_info}
        timer = Timer()
        timer.tic()
        predictions = sess.run(self._predictions, feed_dict=feed_dict)
        timer.toc()
        print('Prediction took {:.3f}s'.format(timer.total_time))
        return predictions

    def final_result(self, preds, im_info):
        rois = {}
        rois_scores = {}
        for branch in self._feat_branches:
            rpn_cls_score = preds[branch]['rpn_cls_score']
            print('>>>>>rpn_cls_score', rpn_cls_score.shape)
            anchors, anchor_length = generate_anchors_pre(rpn_cls_score,
                                                 self._feat_stride[branch], self._anchor_scales[branch],
                                                 self._anchor_ratios)
            anchors = anchors.reshape((-1, 4))
            anchor_length = anchor_length.reshape((-1, ))
            rpn_cls_prob = preds[branch]['rpn_cls_prob']
            rpn_bbox_pred = preds[branch]['rpn_bbox_pred']
            roi, rpn_score = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, self._mode,
                                           self._feat_stride[branch], anchors, self._num_anchors[branch])
            roi = roi.reshape((-1, 5))
            rpn_score = rpn_score.reshape((-1, 1))
            print('>>>>>rois', roi.shape)
            print('>>>>>scores', rpn_score.shape)
            rois[branch] = roi[:, :]
            rois_scores[branch] = rpn_score[:, :]
        if 'M1' in self._feat_branches:
            cls_prob = np.concatenate((rois_scores['M1'], rois_scores['M2'], rois_scores['M3']), axis=0)
            rois = np.concatenate((rois['M1'], rois['M2'], rois['M3']), axis=0)

        # discard M1 branch, only keep M2 and M3 branches to detect medium and large faces
        else:
            print('do not contain M1 branch!!!!')
            cls_prob = np.concatenate((rois_scores['M2'], rois_scores['M3']), axis=0)
            rois = np.concatenate((rois['M2'], rois['M3']), axis=0)
        return cls_prob, rois

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self.gt_kpoints: blobs['gt_kpoints']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self.gt_kpoints: blobs['gt_kpoints']}
        losses, _ = sess.run([self._losses, train_op], feed_dict=feed_dict)
        return losses

    def train_step_old(self, sess, blobs, train_op, module):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self.gt_kpoints: blobs['gt_kpoints']}
        rpn_loss_cls, rpn_loss_box, kpoints_loss, loss, _ = sess.run([self._losses[module]["rpn_cross_entropy"],
                                                        self._losses[module]['rpn_loss_box'],
                                                        self._losses[module]['kpoints_loss'],
                                                        self._losses[module]['total_loss'],
                                                        train_op],
                                                       feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, kpoints_loss, loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self.gt_kpoints: blobs['gt_kpoints']}
        losses, summary, _ = sess.run([self._losses, self._summary_op, train_op], feed_dict=feed_dict)
        return losses, summary

    def train_step_with_summary_old(self, sess, blobs, train_op, module):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self.gt_kpoints: blobs['gt_kpoints']}
        rpn_loss_cls, rpn_loss_box, kpoints_loss, loss_cls, summary, _ = sess.run([self._losses[module]["rpn_cross_entropy"],
                                                                     self._losses[module]['rpn_loss_box'],
                                                                     self._losses[module]['kpoints_loss'],
                                                                     self._losses['total_loss'],
                                                                     self._summary_op,
                                                                     train_op],
                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, kpoints_loss, loss_cls, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self.gt_kpoints: blobs['gt_kpoints']}
        sess.run([train_op], feed_dict=feed_dict)
