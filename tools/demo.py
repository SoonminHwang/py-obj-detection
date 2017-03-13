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
from fast_rcnn.test import im_detect, im_detect_depth
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
CONF_THRESH = 0.01
NMS_THRESH = 0.3


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
    parser.add_argument('--imdb', dest='imdb', help='imdb for demo', 
                        choices=IMDB_NAMES.keys(), default='kitti_test', type=str)
    parser.add_argument('--conf_thres', dest='conf_thres', help='Confidence threshold', 
                        default=CONF_THRESH, type=float)
    parser.add_argument('--nms_thres', dest='nms_thres', help='NMS threshold', 
                        default=NMS_THRESH, type=float)    
    parser.add_argument('--demo_dir', dest='demo_dir',
                        help='A directory that has a few images',
                        default=None, type=str)

    args = parser.parse_args()

    return args


def demo(net, input_names, conf_thres, nms_thres, iter):
    """Detect object classes in an image using pre-computed object proposals."""
    global CLASSES

    # Load the demo image
    # im_file = image_name

    im = cv2.imread(input_names[0])
    fname = os.path.basename(input_names[0])

    # Detect all object classes and regress object bounds
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if 'depth' in cfg.INPUT:
        height, width = im.shape[:2]
	dp = np.load(input_names[1])
	dp[dp == -1] = 0
	#        dp = np.memmap(input_names[1], dtype=np.float32, shape=(height, width))
	#        dp = np.asarray(dp)
        ims = [im, dp]

	#        fig, axes = plt.add_subplots(2,2)

	#        axes[0][0].imshow(im)
	#        axes[0][0].axis('off')
	#        axes[0][1].imshow(dp)
	#        axes[0][1].axis('off')
	#        axes[1][0].imshow(dp)
	#        axes[1][0].axis('off')

	#        plt.savefig('test_align.png', dpi=200)

	#        import ipdb
	#        ipdb.set_trace()


        _t['im_detect'].tic()
        scores, boxes = im_detect_depth(net, ims)
        _t['im_detect'].toc()
    else:
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
    

    plt.figure(1, figsize=(15,10))
    plt.clf()
    axe = plt.gca()
    axe.imshow(im[:,:,(2,1,0)])

    clrs = sns.color_palette("Set2", len(CLASSES))
    for det in results:
        cls_ind, box, score = det[0], det[1:5], det[5]
        if score < 0.8: continue
        clr = clrs[int(cls_ind)]
        rect = plt.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
            fill=False, edgecolor=clr, linewidth=2.5, label=CLASSES[int(cls_ind)])
        axe.add_patch(rect)
        axe.text(box[0], box[1]-2, '{:.3f}'.format(score), 
            bbox=dict(facecolor=clr, alpha=0.5), fontsize=14, color='white')

    axe.axis('off')    
    plt.gca().legend()

    # save_name = os.path.basename(input_names[0])    
    # plt.savefig('[DEMO_iter_%d]' % iter + save_name, dpi=200)  
    plt.savefig('[DEMO_iter_%d]' % iter + fname, dpi=200)  

    return _t
       
if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
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
            
    net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)    
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # imdb
    imdb = get_imdb(IMDB_NAMES[args.imdb])
    print 'imdb: %s' % imdb.name
    global CLASSES
    CLASSES = imdb.classes[:-1]

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)

    if not args.demo_dir:
        # demo from imdb
        for ii in np.random.choice(imdb.num_images, 5):
            im_names = [ imdb.image_path_at(ii) ]
            if 'depth' in cfg.INPUT:
                im_names.append( imdb.depth_path_at(ii) )            
            timer = demo(net, im_names, args.conf_thres, args.nms_thres, model_iter)
            print '[im_detect, {:s}]: {:d}/{:d} {:.3f}s {:.3f}s'.format(os.path.basename(im_names[0]), ii + 1, 
                imdb.num_images, timer['im_detect'].average_time, 
                timer['misc'].average_time)

    else:
        # demo from images
        im_names = glob.glob( os.path.join(args.demo_dir, '*.png') )
        dp_names = glob.glob( os.path.join(args.demo_dir, '*.bin') )
        for im_name, dp_name in zip(im_names, dp_names):            
            timer = demo(net, [im_name, dp_name], args.conf_thres, args.nms_thres, model_iter)
            print '[im_detect, {:s}]: {:.3f}s {:.3f}s'.format(os.path.basename(im_name), 
                timer['im_detect'].average_time, timer['misc'].average_time)      
