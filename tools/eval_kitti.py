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

caffe_path = osp.join(osp.dirname(__file__), '..', '..', '..', '..', 'caffe-latest', 'python')
if caffe_path not in sys.path: sys.path.insert(0, caffe_path)

lib_path = osp.join(osp.dirname(__file__), '..', '..', '..', '..', 'lib')
if lib_path not in sys.path: sys.path.insert(0, lib_path)

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list

# from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.cython_bbox import bbox_overlaps

# from calc_mAP_KITTI import calculate_mAP as calc_mAP

from datasets.factory import get_imdb

from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import seaborn as sns

import ipdb

global CLASSES

#CLASSES = ('__background__', 'Pedestrian', 'Cyclist', 'Car')
#CONF_THRESH = 0.8
CONF_THRESH = 0.01
NMS_THRESH = 0.3

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)

    if len(max_shape) == 3:
        ch = 3
    else:
        ch = 1

    blob = np.zeros((num_images, max_shape[0], max_shape[1], ch),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        if ch == 1: im = im[:,:,np.newaxis]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    # print('cfg.TEST.SCALES: {}'.format(cfg.TEST.SCALES)),

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    # blob /= 255.0

    return blob, np.array(im_scale_factors)

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    
    blobs_out = net.forward(**forward_kwargs)
    
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes


def demo(net, image_name, conf_thres, nms_thres, resDir, iter, bShow=False):
    """Detect object classes in an image using pre-computed object proposals."""
    global CLASSES

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    fname = os.path.basename(image_name)

    # Detect all object classes and regress object bounds
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer(), 'save' : Timer()}

    _t['im_detect'].tic()
    scores, boxes = im_detect(net, im)
    _t['im_detect'].toc()
    
    
    _t['misc'].tic()    
    results = np.zeros((0, 6), dtype=np.float32)
    # Visualize detections for each class
    # for cls_ind, cls in enumerate(CLASSES[1:5]):
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
    
    if bShow:
        plt.figure(1, figsize=(15,10))
        plt.clf()
        axe = plt.gca()
        axe.imshow(im[:,:,(2,1,0)])

        clrs = sns.color_palette("Set2", len(CLASSES))
        for det in results:
            cls_ind, box, score = det[0], det[1:5], det[5]
            if score < 0.8: continue
            clr = clrs[int(cls_ind)]
            rect = plt.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=clr, linewidth=2.5)
            axe.add_patch(rect)
            axe.text(box[0], box[1]-2, '{:.3f}'.format(score), bbox=dict(facecolor=clr, alpha=0.5), fontsize=14, color='white')

        axe.axis('off')
        save_name = os.path.basename(im_file)
        plt.savefig('[DEMO_iter_%d]' % iter + save_name, dpi=200)  

    else:
        _t['save'].tic()
        with open( os.path.join(resDir, fname.split('.')[0] + '.txt'), 'w') as fp:        
            for det in results:
                if len(det) == 0: continue        
                if det[5] < 0.01: continue

                resStr = '{:s} -1 -1 -10 '.format(CLASSES[int(det[0])])                                
                resStr += '{:.2f} {:.2f} {:.2f} {:.2f} '.format(det[1],det[2],det[3],det[4])    # x1 y1 x2 y2
                resStr += '-1 -1 -1 -1000 -1000 -1000 -10 {:.4f}\n'.format(det[5])
                fp.write( resStr )            
        _t['save'].toc()

    return _t
       

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use',
                        default='models/test.prototxt')    
    parser.add_argument('--iter', dest='demo_iter', help='Iteration', default=-1, type=int)    
    parser.add_argument('--conf_thres', dest='conf_thres', help='Confidence threshold', 
                        default=CONF_THRESH, type=float)
    parser.add_argument('--nms_thres', dest='nms_thres', help='NMS threshold', 
                        default=NMS_THRESH, type=float)    
    parser.add_argument('--demo', dest='demo',
                        help='Just test a few images [False]',
                        action='store_true', default=False)    

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    import glob
    cfg_file = glob.glob('*.yml')

    assert len(cfg_file) == 1, 'Too many .cfg files.'
    cfg_from_file(cfg_file[0])
    
        
    prototxt = args.demo_net
    caffemodels = glob.glob('snapshots/*.caffemodel')
    
    import re    
    try:                
        caffemodel = [model for model in caffemodels if int(re.search('iter_(\d+).caffemodel', model).group(1)) == args.demo_iter]
        assert( len(caffemodel) == 1 )
        caffemodel = caffemodel[0]
        print 'Load snapshot: %s' % caffemodel        
        model_iter = args.demo_iter
    except:
        print 'Cannot find iter %d.caffemodel' % args.demo_iter
        
        caffemodel = caffemodels[-1]
        model_iter = int( re.search('iter_(\d+).caffemodel', caffemodel).group(1) )

        print 'Load latest snapshot: %s' % caffemodel                
                
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
        

    #net = caffe.Net(prototxt, caffemodel, caffe.TEST)      # Deprecated warning
    net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)    
    print '\n\nLoaded network {:s}'.format(caffemodel)

    imdb = get_imdb('kitti_2012_val')
    global CLASSES
    CLASSES = imdb.classes 

    # if cfg.TRAIN.DATADRIVEN_ANCHORS:

    #     print '############################################################'
    #     print 'Use Data-driven anchors'
    #     print '############################################################'
  
    #     # Set dataset-specific anchors
    #     anchors = imdb.get_anchors()

    #     proposal_layer_ind = list(net._layer_names).index('proposal')
    #     net.layers[proposal_layer_ind].setup_anchor(anchors)


    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    if not args.demo:
        # Save detections        
        resDir = os.path.join('results', 'iter_%d'%model_iter)
        if not os.path.exists(resDir):
            os.makedirs( resDir )

        for ii in range(imdb.num_images):
            im_name = imdb.image_path_at(ii)
            timer = demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(ii + 1, 
                imdb.num_images, timer['im_detect'].average_time, 
                timer['misc'].average_time, timer['save'].average_time)

        from kitti_evaluate_object import EvalKITTI
        gtDir = os.path.join( cfg.ROOT_DIR, 'data/kitti/annotations/trainval2012/' )
        eval_kitti = EvalKITTI(gtDir, resDir, basePth='')
        eval_kitti.evaluate()
    else:
        for ii in np.random.choice(imdb.num_images, 5):
            im_name = imdb.image_path_at(ii)
            timer = demo(net, im_name, args.conf_thres, args.nms_thres, '', model_iter, True)
            print '[im_detect, {:s}]: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(os.path.basename(im_name), ii + 1, 
                imdb.num_images, timer['im_detect'].average_time, 
                timer['misc'].average_time, timer['save'].average_time)