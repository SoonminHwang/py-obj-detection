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

import _init_paths
from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv

# CLASSES = ('__background__', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Car', 'Van', 'Truck', 'Tram', 'Misc')
CLASSES = ('__background__', 'Pedestrian', 'Cyclist', 'Car')

#CONF_THRESH = 0.8
CONF_THRESH = 0.2
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
    # im_orig -= cfg.PIXEL_MEANS

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
    blob /= 255.0

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

    # ######################################## Draw RPN ########################################

    # scores = net.blobs['rpn_cls_prob'].data.copy()
    # rois = net.blobs['rois'].data.copy()

    # idx = np.where( scores > 0.9 )

    # if len(idx) == 0:
    #     pld.set_trace()

    # proposals = rois[idx[0], 1:]

    # img = net.blobs['data'].data.copy()[0]
    # img = img.transpose( (1,2,0) )    
    # img *= 255.0
    # img = img[:,:,(2,1,0)]


    # plt.figure(11)    
    # plt.clf()
    # plt.title('Proposals, score >= 0.9')
    # ax = plt.gca()
    # ax.imshow(img.astype(np.uint8))
    # for pr in proposals:
    #     ax.add_patch( plt.Rectangle( (pr[0], pr[1]), pr[2]-pr[0], pr[3]-pr[1], fill=False, edgecolor='r'))

    # ########################################################################################### 
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

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""    
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))    

    ax.imshow(im, aspect='equal')
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.2f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    ax.axis('off')

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    
# def demo(net, image_name, conf_thres, nms_thres, resDir):
def demo(net, image_name, conf_thres, nms_thres):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    fname = os.path.basename(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    # Detections
    # dFig, dAx = plt.subplots(len(CLASSES)-1, 1, figsize=(15, 10))   
    dFig, dAx = plt.subplots(4, 1, figsize=(15, 10))   

    plt.ion()
    
    plt.tight_layout()

    # fp = open( os.path.join(resDir, os.path.basename(image_name).split('.')[0]+'.txt'), 'w' )
    results = np.zeros((0, 6), dtype=np.float32)

    # Visualize detections for each class
    # for cls_ind, cls in enumerate(CLASSES[1:5]):
    for cls_ind, cls in enumerate(CLASSES[1:]):                
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thres)
        dets = dets[keep, :]
        results = np.vstack( (results, np.insert(dets, 0, cls_ind, axis=1)) )
        
        vis_detections(im, cls, dets, dAx[cls_ind-1], thresh=conf_thres)        

    # plt.pause(0.1)
    plt.savefig('test.jpg')
    
    # # Ground-truth
    # # gFig, gAx = plt.subplots(len(CLASSES)-1, 1, figsize=(15, 10))
    # gFig, gAx = plt.subplots(4, 1, figsize=(15, 10))

    # # for ii in range(len(CLASSES)-1):
    # for ii in range(4):
    #     gAx[ii].imshow(im[:, :, (2, 1, 0)])
    
    # annNm = os.path.dirname(im_file) + '/' + fname.split('.')[0] + '.txt'


    # np.set_printoptions(precision=2)

    # annotations = []
    # with open(annNm, 'r') as f:
    #     lines = f.readlines()
        
    #     for line in lines:
    #         d = line.split(' ')
    #         nums = [float(num) for num in d[1:]]
    #         clsStr = d[0]
    #         trunc, occ, alpha = nums[:3]
    #         left, top, right, bottom = nums[3:7]            
    #         #bbox    = [left, top, right-left+1, bottom-top+1]
    #         bbox    = [left, top, right, bottom]
            
    #         cls_ind = [ind for ind, clsNm in enumerate(CLASSES[1:]) if clsStr == clsNm]            
    #         annotations.append( bbox + cls_ind )

    # try:
    #     for cls_ind in range(len(CLASSES)-1):
    #         gt_boxes = np.asarray([box for box in annotations if box[-1] == cls_ind])
    #         dt_boxes = results[results[:,0] == cls_ind+1, :]

    #         if len(gt_boxes) == 0: continue

    #         overlaps = bbox_overlaps( np.ascontiguousarray(gt_boxes, dtype=np.float), np.ascontiguousarray(dt_boxes[:,1:], dtype=np.float))
    #         argmax_overlaps = overlaps.argmax(axis=1)
    #         max_overlaps = overlaps[np.arange(len(gt_boxes)), argmax_overlaps]

    #         gt_argmax_overlaps = overlaps.argmax(axis=0)
    #         gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    #         gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    #         for ii, gt_box in enumerate(gt_boxes):
    #             if gt_max_overlaps[ii] >= 0.5:
    #                 clr = 'r'
    #                 ovlStr = '{:.2f}'.format(gt_max_overlaps[ii])
    #             else:
    #                 clr = 'b'
    #                 ovlStr = ''

    #             gAx[cls_ind].add_patch(
    #                 plt.Rectangle( (gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1], fill=False,
    #                     edgecolor=clr, linewidth=3)
    #                 )
    #             gAx[cls_ind].text(gt_box[0], gt_box[1]-2, ovlStr, color='white', 
    #                 bbox={'facecolor': clr, 'alpha':0.5})

    #             #gAx[cls_ind].text(bbox[0], bbox[1]-2, '{:s}'.format(CLASSES[cls_ind+1]), color='white', 
    #                 #bbox={'facecolor': clr, 'alpha':0.5})
    # except:
    #     pdb.set_trace()

    # plt.show()
    # plt.draw()        
    # plt.pause(1)
    # plt.savefig(os.path.join(cfg.DATA_DIR, 'demo', '[Result]' + fname))
                
    for ii in range(len(results)):
        print('[%d] %8.2f, %8.2f, %8.2f, %8.2f\t%.4f'%
            (results[ii][0], results[ii][1], results[ii][2], results[ii][3], results[ii][4], results[ii][5]))

    print('# of results: {} (>= {:.2f}: {} detections)'.format(
        len(results), conf_thres, len([1 for r in results if r[-1] >= conf_thres])))

    print('')

    raw_input("Press enter to continue")
       

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [*.prototxt]')
    parser.add_argument('--caffemodel', dest='caffemodel', help='Trained weights [*.caffemodel]')    

    parser.add_argument('--conf_thres', dest='conf_thres', help='Confidence threshold', 
                        default=CONF_THRESH, type=float)
    parser.add_argument('--nms_thres', dest='nms_thres', help='NMS threshold', 
                        default=NMS_THRESH, type=float)
    parser.add_argument('--method', dest='method_name', help='Algorithm name', type=str)
    parser.add_argument('--evalDir', dest='eval_dir', 
			help='For evaluation, please specify evaluation directory',
			default='', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])

    prototxt = args.demo_net
    caffemodel = args.caffemodel
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        # cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    #            'KITTI_003313.png', 'KITTI_000023.png', 'KITTI_000211.png', 'KITTI_001443.png' ] # KITTI Test set

    # import datetime
    # nowStr = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    # resDir = os.path.join(cfg.DATA_DIR, 'kitti', 'evaluation', 'results', nowStr + '_' + args.method_name)
    # if not os.path.exists(resDir):
    #     os.makedirs( resDir )

    if args.eval_dir is not '':
        import glob
        im_names = glob.glob( os.path.join(args.eval_dir, '*.png') )
        print( 'Get test images from {}'.format(args.eval_dir) )

        for im_name in im_names:            
            demo(net, im_name, args.conf_thres, args.nms_thres)
            # demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
    else:
        #im_names = ['KITTI_000017.png']
        im_names = ['KITTI_003686.png', 'KITTI_003687.png',
       		'KITTI_003690.png', 'KITTI_003694.png', 'KITTI_003709.png', 'KITTI_003683.png', 'KITTI_003684.png' ] # KITTI Test set]
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/demo/{}'.format(im_name)
            im_name = os.path.join( cfg.DATA_DIR, 'demo', im_name )
            demo(net, im_name, args.conf_thres, args.nms_thres)
            # demo(net, im_name, args.conf_thres, args.nms_thres, resDir)
     
    #plt.show()
