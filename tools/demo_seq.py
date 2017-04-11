#!/usr/bin/env python
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# Modified by Soonmin Hwang
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
Modified to enhance flexivility for trained model selection and dataset

See README.md for installation instructions before running.
"""

#import _init_paths

import sys
import os.path as osp

caffe_path = osp.join(osp.dirname(__file__), '..', 'caffe-latest', 'python')
if caffe_path not in sys.path: sys.path.insert(0, caffe_path)

lib_path = osp.join(osp.dirname(__file__), '..', 'lib')
if lib_path not in sys.path: sys.path.insert(0, lib_path)

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, im_detect_depth, im_detect_edge
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from datasets.factory import get_imdb
import seaborn as sns
import glob

IMDB_NAMES = {  'kitti_val': 'kitti_2012_val', 
                'kitti_test': 'kitti_2012_test',
                'voc07_test': 'voc_2007_test'}

global CLASSES
CONF_THRESH = 0.6
NMS_THRESH = 0.3


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--dir', dest='exp_dir', help='Log directory', type=str)
    parser.add_argument('--model', dest='model', help='Network model', type=str)
    parser.add_argument('--iter', dest='iter', help='Iteration', default=-1, type=int)    
    parser.add_argument('--seqDir', dest='seq_dir', help='Sequence dir', type=str)    
    parser.add_argument('--conf_thres', dest='conf_thres', help='Confidence threshold', 
                        default=CONF_THRESH, type=float)
    parser.add_argument('--nms_thres', dest='nms_thres', help='NMS threshold', 
                        default=NMS_THRESH, type=float)    
    
    args = parser.parse_args()
    return args


def demo(net, input_names, gts, conf_thres, nms_thres, iter, prefix=''):
    """Detect object classes in an image using pre-computed object proposals."""
    global CLASSES

    # Load the demo image    
    im = cv2.imread(input_names[0])
    fname = os.path.basename(input_names[0])

    # Detect all object classes and regress object bounds
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if 'edge' in cfg.INPUT:
        input_data = cv2.imread(input_names[1], -1)        
        input_data = input_data.astype(np.float32) - 128.0

        ims = [im, input_data]        

        _t['im_detect'].tic()
        scores, boxes = im_detect_edge(net, ims)
        _t['im_detect'].toc()
    else:
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im)
        _t['im_detect'].toc()
    

    _t['misc'].tic()    
    results = np.zeros((0, 6), dtype=np.float32)

    for cls_ind, cls in enumerate(CLASSES[1:]):    
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
        # CPU NMS is much faster than GPU NMS when the number of boxes
        # is relative small (e.g., < 10k)
        # TODO(rbg): autotune NMS dispatch
        keep = nms(dets, nms_thres, force_cpu=True)
        dets = dets[keep, :]
        results = np.vstack( (results, np.insert(dets, 0, cls_ind, axis=1)) )        
    _t['misc'].toc()  
    

    plt.figure(1, figsize=(15,10))
    plt.clf()
    axe = plt.gca()
    axe.imshow(im[:,:,(2,1,0)])

    # clrs = sns.color_palette("hsv", len(CLASSES))
    # for gt in gts:
    #     cls_ind, box = gt[0], gt[1:5]        
    #     clr = clrs[int(cls_ind)]
    #     rect = plt.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
    #         fill=False, edgecolor=clr, linewidth=2.5, label=CLASSES[int(cls_ind)])
    #     axe.add_patch(rect)
        
    bLabel = np.zeros(len(CLASSES))

    clrs = sns.color_palette("Set2", len(CLASSES))
    for det in results:
        cls_ind, box, score = det[0], det[1:5], det[5]
        if score < conf_thres: continue
        clr = clrs[int(cls_ind)]
        if bLabel[int(cls_ind)] == 0:
            rect = plt.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                fill=False, edgecolor=clr, linewidth=2.5, label=CLASSES[int(cls_ind)])
            bLabel[int(cls_ind)] = 1
        else:
            rect = plt.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                fill=False, edgecolor=clr, linewidth=2.5)

        axe.add_patch(rect)
        axe.text(box[0], box[1]-2, '{:.3f}'.format(score), 
            bbox=dict(facecolor=clr, alpha=0.5), fontsize=14, color='white')

    axe.axis('off')    
    plt.gca().legend()
    
    plt.tight_layout()

    prefix = prefix + '_' if prefix is not '' else ''
    saveDir = os.path.dirname(input_names[0]).replace('image_02', 'results_02')
    saveDir = os.path.join( saveDir, prefix + 'iter_%06d' % iter )

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    plt.savefig( os.path.join(saveDir, fname), dpi=200 )

    return _t

# def read_gts(file):




if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
    expDir = args.exp_dir
    model = args.model

    cfg_from_file( os.path.join(expDir, 'faster_rcnn_end2end_kitti_%s.yml' % model ))
            
    caffemodel = os.path.join(expDir, 'snapshots/%s_iter_%d.caffemodel' % (model.split('_')[0].lower(), args.iter))
    prototxt = os.path.join(expDir, 'models/network.prototxt')

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
            
    net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)    
    print '\n\nLoaded network {:s}'.format(caffemodel)

    global CLASSES
    CLASSES = ['__background__', 'Pedestrian', 'Cyclist', 'Car']
    
    files = glob.glob( os.path.join(args.seq_dir, '*.png') )
    files.sort()
    for ii, file in enumerate(files):
        # gts = read_gts( file.replace('image_02', 'label_02').replace('png', 'txt') )
        im_names = [ file ]
        if 'edge' in cfg.INPUT:
            im_names.append( file.replace('image_02', 'edge_02') )

        gts = []
        timer = demo(net, im_names, gts, args.conf_thres, args.nms_thres, args.iter, model)

        print '[im_detect, {:s}]: {:d}/{:d} {:.3f}s {:.3f}s'.format(os.path.basename(file), ii + 1, 
                len(files), timer['im_detect'].average_time, 
                timer['misc'].average_time)
