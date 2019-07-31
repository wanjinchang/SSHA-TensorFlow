#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tools._init_paths
from lib.model.config import cfg
from lib.model.test import im_detect, im_detect_bbox_kpoints
from lib.model.nms_wrapper import nms

from lib.utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import time

from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
from lib.nets.mobilenet_v1 import mobilenetv1
from lib.nets.darknet53 import Darknet53
from lib.nets.mobilenet_v2.mobilenet_v2 import mobilenetv2

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__', 'face')

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NETS = {'vgg16': ('vgg16_ssh_iter_460000.ckpt',), 'res101': ('res101_ssh_iter_110000.ckpt',),
        'mobile': ('mobile_ssh_iter_400000.ckpt',), 'res50': ('res50_ssh_iter_310000.ckpt',),
        'darknet53': ('darknet53_ssh_iter_400000.ckpt',), 'mobile_v2': ('mobile_v2_ssh_iter_320000.ckpt',)}
DATASETS= {'wider_face': ('wider_face_train',)}

def vis_detections(im, class_name, dets, result_file, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='blue', linewidth=2)
            )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.2f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(result_file)

def video_demo(sess, net, image):
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, _ = im_detect_bbox_kpoints(sess, net, image)
    # scores, boxes, points = im_detect(sess, net, image)
    # print("scores:", scores.shape)  --> (n, 1)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3

    inds = np.where(scores[:, 0] > CONF_THRESH)[0]
    scores = scores[inds, 0]
    boxes = boxes[inds, :]
    # points = points[inds, :]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    # dets = np.hstack((boxes, scores[:, np.newaxis], points)).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets
    # vis_detections(image, CLASSES[1], dets, thresh=CONF_THRESH)

def cv2_vis(im, class_name, dets, kpoints):
    """Draw detected bounding boxes using cv2."""
    # inds = np.where(dets[:, -1] >= thresh)[0]
    # im = im[:, :, ::-1].copy()
    if dets.shape[0] != 0:
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            # score = dets[i, -1]
            score = dets[i, 4]
            # points = dets[i, 5:]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            cv2.rectangle(im, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0,  0, 255), 2)
            for k in range(kpoints.shape[1] // 2):
                cv2.circle(im, (int(kpoints[i][2*k]),int(int(kpoints[i][2*k+1]))), 1, (0, 0, 255), 2)
    cv2.imshow("demo", im)
    # cv2.imwrite(result_file, im)
    cv2.waitKey(0)

def demo(sess, net, img_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    once_time = 0

    im = cv2.imread(img_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, kpoints = im_detect_bbox_kpoints(sess, net, im)
    timer.toc()
    once_time = timer.total_time
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.85
    NMS_THRESH = 0.3

    inds = np.where(scores[:, 0] > CONF_THRESH)[0]
    scores = scores[inds, 0]
    boxes = boxes[inds, :]
    kpoints = kpoints[inds, :]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    kpoints = kpoints[keep, :]
    print('>>>>>num_faces:', dets.shape[0])
    for i in range(dets.shape[0]):
        print('>>>>>face width{}, height{}'.format(int(dets[i][2]) - int(dets[i][0]), int(dets[i][3]) - int(dets[i][1])))
    cv2_vis(im, CLASSES[1], dets, kpoints)

    return once_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow SSH demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101 res50 mobile]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [wider_face]',
                        choices=DATASETS.keys(), default='wider_face')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    ### vgg16 default_group_20190111
    ### mobile_v1 default_group_fixed_3_layers_20181220
    tfmodel = os.path.join('/home/oeasy/PycharmProjects/tf-ssh_modify/output', demonet, DATASETS[dataset][0], 'default_bbx_kp_0617',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.07
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        print('ssh base_network is vgg16')
        net = vgg16()
    elif demonet == 'res101':
        print('ssh base_network is resnet101')
        net = resnetv1(num_layers=101)
    elif demonet == 'res50':
        print('ssh base_network is resnet50')
        net = resnetv1(num_layers=50)
    elif demonet == 'mobile':
        print('ssh base_network is mobilenet_v1')
        net = mobilenetv1()
    elif demonet == 'darknet53':
        print('ssh base_network is darknet53')
        net = Darknet53()
    elif demonet == 'mobile_v2':
        print('ssh base_network is mobilenet_v2')
        net = mobilenetv2()
    else:
        raise NotImplementedError

    # SSH original anchors scales
    anchor_scales = {"M1": [1, 2], "M2": [4, 8], "M3": [16, 32]}
    # add anchors
    # anchor_scales = {"M1": [0.5, 1.0, 1.5, 2.0], "M2": [0.5, 1.0, 1.5, 2.0, 4, 8], "M3": [0.5, 1.0, 1.5, 2.0, 16, 32]}

    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales=anchor_scales)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['001150.jpg', 'demo.jpg', '004545.jpg', '1.jpeg', '2.jpeg',
    #             '3.jpeg', '4.jpeg', '5.jpeg', '6.jpeg', '7.jpeg', '8.jpeg', '9.jpeg', '10.jpeg',
    #             '11.jpeg', '12.jpeg', '13.jpeg', '14.jpeg', '15.jpeg', '16.jpeg', '17.jpeg', '18.jpeg',
    #             '19.jpeg', '20.jpeg', '21.jpeg', '22.jpeg', '23.jpeg', '24.jpeg', '25.jpeg', '26.jpeg',
    #             '27.jpeg', '28.jpeg', '29.jpeg', '30.jpg', '31.jpg', '32.jpg', '33.jpeg', '34.jpeg', '35.jpeg',
    #             '36.jpeg', '37.jpeg', '38.jpeg', '39.jpeg', '40.jpeg', 'img_414.jpg', 'img_423.jpg', 'img_17676.jpg']


    # images_dir = os.path.join(cfg.DATA_DIR, 'wider_eval')
    images_dir = '/home/oeasy/PycharmProjects/oeasy_face_lib_0612.git/example/save_org'
    im_names = os.listdir(images_dir)
    print('>>>>', im_names)
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        img_path = os.path.join(images_dir, im_name)
        demo(sess, net, img_path)

    ############### test inference time ###################################
    # total_time = 0
    # inference = 0
    # for i in range(10000):
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for data/demo/{} {}'.format('oeasy_2.jpg', str(i)))
    #     img_path = os.path.join(cfg.DATA_DIR, 'demo', '36.jpeg')  # 15  18
    #     once_time, inference_time = demo(sess, net, img_path)
    #     if i > 10:
    #         total_time += once_time
    #         inference += inference_time
    # avg_total_time = total_time / (10000 - 10)
    # avg_inference_time = inference / (10000 - 10)
    # print('the average total time for {}_ssh took {:.3f}s'.format(demonet, avg_total_time))
    # print('the average inference time for {}_ssh took {:.3f}s'.format(demonet, avg_inference_time))

    # while True:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for data/demo/{}'.format('19.jpeg'))
    #     demo(sess, net, '19.jpeg')
        # plt.show()
    # demo(sess, net, 'oeasy_1.jpg')
    # plt.show()
    # plt.show()
    #


    ################################################ video test demo #################################
    # videopath = "./video_test.avi"
    # videopath = "/home/oeasy/ch01_20190306074318.mp4"
    # video_capture = cv2.VideoCapture(videopath)
    # # video_capture.set(3, 340)
    # # video_capture.set(4, 480)
    # while True:
    #     # fps = video_capture.get(cv2.CAP_PROP_FPS)
    #     t1 = cv2.getTickCount()
    #     ret, frame = video_capture.read()
    #     # h, w, _ = frame.shape
    #     # print("video height: %s & width: %s" % (h, w))   # video height: 240 & width: 320
    #     if ret:
    #         image = np.array(frame)
    #         detetctions = video_demo(sess, net, image)
    #         print('num_faces:', detetctions.shape[0])
    #         t2 = cv2.getTickCount()
    #         t = (t2 - t1) / cv2.getTickFrequency()
    #         fps = 1.0 / t
    #         for i in range(detetctions.shape[0]):
    #             bbox = detetctions[i, :4]
    #             score = detetctions[i, 4]
    #             corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    #             # if score > thresh:
    #             cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
    #                           (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
    #             cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.5,
    #                         (0, 0, 255), 2)
    #         cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     (255, 0, 255), 2)
    #         cv2.imshow("", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         print('device not find')
    #         break



