# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob, prep_im_for_blob_randscale

from transform.image_transform import _flip, _crop_resize, _gamma_correction
# from scipy import misc

def get_minibatch(roidb, num_classes, randScale):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    # im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    # blobs = {'data': im_blob}

    input_blobs = _get_input_blob(roidb, random_scale_inds, randScale)
    im_scales = input_blobs[-1]

    # blobs = { item.key(): item.value() for item in input_blobs[0] }
    blobs = input_blobs[0]
    
    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        
        # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes = np.empty((len(gt_inds), 7), dtype=np.float32)        # Add occ, trunc

        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        gt_boxes[:, 5] = roidb[0]['gt_occ'][gt_inds]
        gt_boxes[:, 6] = roidb[0]['gt_trunc'][gt_inds]

        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[blobs['image'].shape[2], blobs['image'].shape[3], im_scales[0]]],
            dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    # For debug visualizations
    # _vis_minibatch_rpn(input_blobs, roidb)
    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_input_blob(roidb, scale_inds, randScale):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    # im_scales = []
    # blob = {'scales': []}
    blob = {}
    for i in xrange(num_images):

        n_input_types = len(roidb[i]['input'])
                
        width = roidb[i]['width']
        height = roidb[i]['height']
        
        N = len(cfg.TRAIN.USE_AUGMENTATION.IM_SCALES)    
        im_scale = cfg.TRAIN.USE_AUGMENTATION.IM_SCALES[ npr.choice(N, 1)]
        
        for j in xrange(n_input_types):
            input_type = roidb[i]['input'][j].keys()[0]
            input_file = roidb[i]['input'][j].values()[0]


            if input_type == 'image':
                input_data = cv2.imread(input_file, -1)                

            elif input_type == 'depth':

                ## DispFlowNet
                with open(input_file, 'rb') as f:
                    assert( f.readline() == 'float\n' )
                    assert( int(f.readline()) == 3 )
                    w = int(f.readline())
                    h = int(f.readline())
                    c = int(f.readline())

                    import array
                    disp = array.array('f')
                    disp.fromfile(f, h*w*c)

                    if c != 1:
                        disp = -1 * np.asarray(disp, dtype=np.float32).reshape(h, w, c)
                    else:
                        disp = -1 * np.asarray(disp, dtype=np.float32).reshape(h, w)

                    input_data = roidb[i]['focal'] * roidb[i]['baseline'] / disp

                    # [yy, xx] = np.meshgrid(np.arange(width), np.arange(height))
                    # yy = yy.astype(np.float32) / height - 0.5
                    # xx = xx.astype(np.float32) / width - 0.5

                    # max_depth = 100.0
                    # input_data = input_data / max_depth - 0.5 
                    # input_data = np.stack( (input_data, xx, yy), axis=2 )
            elif input_type == 'edge':
                input_data = cv2.imread(input_file, -1)
                input_data = input_data.astype(np.float32) / 255.0 - 0.5

            else:
                raise NotImplementedError

            # # Load 16-bit uint png image  
            # input_data = cv2.imread(input_file, -1)
            # # input_data = misc.imread( input_file )
            
            # if input_data.dtype == np.uint16:
            #     # From kitti/flow2015/devkit/matlab/disp_read.m,
            #     input_data = input_data.astype(np.float32) / 256.0        

            #     if cfg.USE_METRIC_DEPTH:
            #         mask = input_data == 0.0
            #         input_data[ mask ] = -1
            #         input_data = roidb[i]['focal'] * roidb[i]['baseline'] / input_data
            #         input_data[ mask ] = 0.0

            #         [yy, xx] = np.meshgrid(np.arange(width), np.arange(height))
            #         yy = yy.astype(np.float32) / height - 0.5
            #         xx = xx.astype(np.float32) / width - 0.5

            #         max_depth = 100.0
            #         input_data = input_data / max_depth - 0.5 
            #         input_data = np.stack( (input_data, xx, yy), axis=2 )


            # if input_file.endswith('.png'):
            #     input_data = cv2.imread(input_file)
            # else:                
            #     input_data = np.load(input_file)
            #     input_data[input_data == -1] = 0

            #     # input_data = np.memmap(input_file, dtype=np.float32, shape=(height, width))
            #     # input_data = np.asarray(input_data)

            if roidb[i]['flipped']:
                input_data = _flip(input_data)
            if roidb[i]['gamma'] and input_type != 'depth':
                input_data = _gamma_correction(input_data)
            if roidb[i]['crop'] is not None:
                input_data = _crop_resize(input_data, roidb[i]['crop'])

            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            mean_pixels = cfg.PIXEL_MEANS if input_type == 'image' else 0.0

            # input_data, im_scale = prep_im_for_blob(input_data, mean_pixels, target_size,
                                            # cfg.TRAIN.MAX_SIZE)
            if randScale:
                input_data = prep_im_for_blob_randscale(input_data, mean_pixels, im_scale)
            else:
                input_data, im_scale = prep_im_for_blob(input_data, mean_pixels, target_size,
                                            cfg.TRAIN.MAX_SIZE)
                        
            # im_scales.append(im_scale)
            # blob.append( { input_type: input_data } )
            # processed_ims.append(input_data)

            # Create a blob to hold the input images            
            blob[input_type] = im_list_to_blob([input_data])

    # blob['scales'] = [ im_scale ]
    return blob, [im_scale]

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch_rpn(blobs, roidb):
    import matplotlib.pyplot as plt
    
    plt.ion()

    im = blobs[0]['image'][0, :, :, :].transpose((1,2,0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2,1,0)]
    im = im.astype(np.uint8)

    boxes = blobs[0]['gt_boxes']

    plt.figure(1)
    plt.clf()
    plt.imshow(im)
    axe = plt.gca()

    import seaborn as sns
    clrs = sns.color_palette("Set2", 5)    
    for box in boxes:
        clr = clrs[int(box[-1])]
        axe.add_patch( 
            plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                fill=False, edgecolor=clr, linewidth=3, label='%s' % box[-1])
        )

    axe.axis('off')
    plt.legend()
    plt.show()
    plt.pause(1)
    

    import ipdb
    ipdb.set_trace()

    plt.ioff()



def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
