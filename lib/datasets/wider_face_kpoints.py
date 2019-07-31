#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: wider_face_kpoints.py
@time: 18-6-26 上午9:38
@desc:
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.datasets.imdb import imdb
import lib.datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import lib.utils.cython_bbox
from PIL import Image
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from lib.model.config import cfg


class wider_face(imdb):
    def __init__(self, image_set, use_diff=False):
        name = 'wider_face' + '_' + image_set
        if use_diff:
            name += '_diff'
        imdb.__init__(self, name)
        self._image_set = image_set
        self._data_path = self._get_default_path()
        # self._devkit_path = self._get_default_path()
        # self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'face')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'

        self._fp_bbox_map = {}
        self._fp_kpoint_map = {}
        self._image_index = []
        image_set_file = os.path.join(self._data_path, 'Annotations',
                                      'wider_face_' + self._image_set + '_bbx_kp.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            annotations = f.readlines()
        if self._image_set == 'train':
            count = 0
            while count < len(annotations):
                name = str(annotations[count]).rstrip()
                self._image_index.append(name)
                self._fp_bbox_map[name] = []
                self._fp_kpoint_map[name] = []
                # self._ind_kpoint_map[name] = []
                count += 1
                n_anno = int(annotations[count])
                for i in range(n_anno):
                    count += 1
                    bbox = annotations[count].split(' ')[0:4]
                    kpoint = annotations[count].split(' ')[4:]

                    bbox = [int(round(float(x))) for x in bbox]
                    kpoint = [int(round(float(x))) for x in kpoint]
                    x1 = max(0, bbox[0])
                    y1 = max(0, bbox[1])
                    bbox[0] = x1
                    bbox[1] = y1
                    self._fp_bbox_map[name].append(bbox)
                    self._fp_kpoint_map[name].append(kpoint)
                count += 1
        else:
            for annotation in annotations:
                annotation = annotation.strip().split(' ')
                name = annotation[0]
                self._image_index.append(name)
                self._fp_bbox_map[name] = []
                self._fp_kpoint_map[name] = []
                # image_index.append(annotation[0])
                bboxes = list(map(float, annotation[1:]))
                num_objs = int(len(bboxes) / 4)
                for idx in range(num_objs):
                    xmin = float(bboxes[4 * idx])
                    ymin = float(bboxes[4 * idx + 1])
                    xmax = float(bboxes[4 * idx + 2])
                    ymax = float(bboxes[4 * idx + 3])
                    bbox = [xmin, ymin, xmax, ymax]
                    bbox = [int(round(float(x))) for x in bbox]
                    self._fp_bbox_map[name].append(bbox)
        # Default to roidb handler
        # self._image_index = [key for key, _ in self._fp_bbox_map.items()]

        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': use_diff,
                       'matlab_eval': False,
                       'rpn_file': None}

        # assert os.path.exists(self._devkit_path), \
        #   'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        if index.endswith('.jpg'):
            image_path = os.path.join(self._data_path, 'images', index)
        else:
            image_path = os.path.join(self._data_path, 'images', index+self._image_ext)
        # image_path = self._data_path + '/images/' + index.split(' ')[1] + self._image_ext
        # print('>>>>>', image_path)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index_and_annotations(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /Annotations/val.txt
        image_index = []
        bboxes = []
        landmarks_flag = []
        landmarks = []
        image_set_file = os.path.join(self._data_path, 'Annotations',
                                      'wider_face_' + self._image_set + '_bbx_kp.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            # image_index = [x.strip() for x in f.readlines()]
            # image_index = [x.strip().split(' ')[0] for x in f.readlines()]
            annotations = f.readlines()
        for annotation in annotations:
            annotation = annotation.strip().split(' ')
            image_index.append(annotation[0])
            with_landmarks = int(annotation[1])
            # print('annotation', annotation[0])
            # print('>>>>>flag', flag)
            # if landmark_flag == 1 --> with landmarks else without landmarks
            ### face with landmarks(only one face annotation)
            if with_landmarks:
                bbox = list(map(float, annotation[2:6]))
                landmark = list(map(float, annotation[6:]))
            else:
                bbox = list(map(float, annotation[2:]))
                landmark = [0.0] * 10
            bboxes.append(bbox)
            landmarks.append(landmark)
            landmarks_flag.append(with_landmarks)
        return image_index, bboxes, landmarks_flag, landmarks

    def _get_default_path(self):
        """
        Return the default path where Wider Face is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'WIDER/WIDER_' + self._image_set)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_bbox_kp_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        # gt_roidb = [self._load_widerface_annotation(index)
        #             for index in self.image_index]
        # gt_roidb = self._load_widerface_annotation()
        gt_roidb = self._load_widerface_bbx_kp_annotation()
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_widerface_annotation(self):
        """
        Load image and bounding boxes info from txt file in the WIDER_Face annotation txt file
        format.
        """
        gt_roidb = []
        for index, bbox in enumerate(self.bboxes):
            num_objs = int(len(bbox) / 4)
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)

            # Load object bounding boxes into a data frame.
            for idx in range(num_objs):
                xmin = float(bbox[4 * idx])
                ymin = float(bbox[4 * idx + 1])
                xmax = float(bbox[4 * idx + 2])
                ymax = float(bbox[4 * idx + 3])
                if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                    continue
                boxes[idx, :] = [xmin, ymin, xmax, ymax]
                # only face is the object, one class
                cls = int(1)
                gt_classes[idx] = cls
                overlaps[idx, cls] = 1.0
                seg_areas[idx] = (xmax - xmin + 1) * (ymax - ymin + 1)

            overlaps = scipy.sparse.csr_matrix(overlaps)
            bbox_object = {'boxes': boxes,
                           'gt_classes': gt_classes,
                           'gt_overlaps': overlaps,
                           'flipped': False,
                           'seg_areas': seg_areas}
            gt_roidb.append(bbox_object)

        return gt_roidb

    def _load_widerface_bbx_kp_annotation(self):
        """
        Load image and bounding boxes info from txt file in the WIDER_Face annotation txt file
        format.
        """
        gt_roidb = []
        for fp in self._image_index:
            boxes = np.zeros((len(self._fp_bbox_map[fp]), 4), np.float)
            # points = np.zeros((len(self._fp_kpoint_map[fp]), 11), np.float)
            points = np.zeros((len(self._fp_bbox_map[fp]), 11), np.float)
            gt_classes = np.zeros((len(self._fp_bbox_map[fp])), np.int32)
            overlaps = np.zeros((len(self._fp_bbox_map[fp]), self.num_classes), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((len(self._fp_bbox_map[fp])), dtype=np.float32)

            ix = 0
            if fp.endswith('.jpg'):
                imsize = Image.open(os.path.join(self._data_path, 'images' ,fp)).size
            else:
                imsize = Image.open(os.path.join(self._data_path, 'images', fp+'.jpg')).size
            for bbox in self._fp_bbox_map[fp]:
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = min(imsize[0], bbox[2])
                ymax = min(imsize[1], bbox[3])
                if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                    continue
                boxes[ix, :] = [xmin, ymin, xmax, ymax]
                if self._image_set == 'train':
                    box_ind = self._fp_bbox_map[fp].index(bbox)
                    points[ix, :] = np.array(self._fp_kpoint_map[fp][box_ind], np.float32)

                cls = int(1)
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (xmax - xmin + 1) * (ymax - ymin + 1)
                ix += 1

            overlaps = scipy.sparse.csr_matrix(overlaps)
            bbox_kp_object = {'boxes': boxes,
                              'points': points,
                              'gt_classes': gt_classes,
                              'gt_overlaps': overlaps,
                              'flipped': False,
                              'seg_areas': seg_areas}
            gt_roidb.append(bbox_kp_object)
        return gt_roidb

    def write_detections(self, all_boxes, output_dir='./output/'):

        print('Writing the detections to text files: {}...'.format(output_dir), end='')
        for i in range(len(self._image_paths)):
            img_path = self._image_paths[i]

            img_name = os.path.basename(img_path)
            img_dir = img_path[:img_path.find(img_name) - 1]

            txt_fname = os.path.join(output_dir, img_dir, img_name.replace('jpg', 'txt'))

            res_dir = os.path.join(output_dir, img_dir)
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)

            with open(txt_fname, 'w') as f:
                f.write(img_path + '\n')
                f.write(str(len(all_boxes[1][i])) + '\n')
                for det in all_boxes[1][i]:
                    f.write('%d %d %d %d %g \n' % (
                        int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]),
                        det[4]))
        print('Done!')

    def evaluate_detections(self, all_boxes, output_dir='./output/',method_name='SSH'):
        detections_txt_path = os.path.join(output_dir, 'detections')
        self.write_detections(all_boxes, detections_txt_path)

        print('Evaluating detections using official WIDER toolbox...')
        path = os.path.join(os.path.dirname(__file__),
                            '..', 'wider_eval_tools')
        eval_output_path = os.path.join(output_dir, 'wider_plots')
        if not os.path.isdir(eval_output_path):
            os.mkdir(eval_output_path)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; '
        cmd += 'wider_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(detections_txt_path, method_name, eval_output_path)
        print('Running:\n{}'.format(cmd))
        subprocess.call(cmd, shell=True)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from lib.datasets.pascal_voc import pascal_voc

    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
