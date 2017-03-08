# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

# Added for new features: store solverstate & unnormalized params, visualize
import cPickle
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
import seaborn as sns


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb_train, roidb_val, imdb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.imdb = imdb

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb_train)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
            if len(self.solver.test_nets) > 0:
                self.solver.test_nets[0].copy_from(pretrained_model)
                self.solver.test_nets[0].share_with(self.solver.net)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb_train)
        if len(self.solver.test_nets) > 0:
            self.solver.test_nets[0].layers[0].set_roidb(roidb_val)

    def visualize(self, net, filename):        
        
        blobs_out = net.forward()

        try:
            im = net.blobs['data'].data[0].copy()
        except:
            im = net.blobs['image'].data[0].copy()

        im = im.transpose((1,2,0))  # ch x h x w -> h x w x ch
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]

        im_scale = float(cfg.TEST.SCALES[0]) / float(min(im.shape[:2]))

        if cfg.TEST.HAS_RPN:
            # assert len(im_scale) == 1, "Only single-image batch implemented"
            rois = net.blobs['rois'].data.copy()
            # unscale back to raw image space
            boxes = rois[:, 1:5] / im_scale
        elif cfg.DEDUP_BOXES > 0:
            raise NotImplementedError
            # When mapping from image ROIs to feature map ROIs, there's some aliasing
            # (some distinct image ROIs get mapped to the same feature ROI).
            # Here, we identify duplicate feature ROIs, so we only compute features
            # on the unique subset.
            # v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            # hashes = np.round(net.blobs['rois'].data.copy() * cfg.DEDUP_BOXES).dot(v)
            # _, index, inv_index = np.unique(hashes, return_index=True,
            #                                 return_inverse=True)
            # rois = net.blobs['rois'][index, :]
            # boxes = boxes[index, :]

        # use softmax estimated probabilities       
        scores = net.blobs['cls_score'].data.copy()
        scores = np.exp(scores)
        scores_sum = np.sum(scores, axis=1)[:,np.newaxis]
        scores /= scores_sum

        # scores = scores.max(axis=1)
        # scores = blobs_out['cls_score']

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas            
            try:
                box_deltas = net.blobs['bbox_pred'].data.copy()
            except:
                box_deltas = net.blobs['bbox_pred_depth'].data.copy()
                
            box_deltas = box_deltas * self.bbox_stds + self.bbox_means
            # box_deltas = blobs_out['bbox_pred']
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, im.shape)
        else:
            print '[Warning] Bounding-box regression is not applied at test phase.'
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        # if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        #     # Map scores and predictions back to the original set of boxes
        #     scores = scores[inv_index, :]
        #     pred_boxes = pred_boxes[inv_index, :]

        # Post-processing
        imdb = self.imdb
        thresh = 0.8
        
        clrs = sns.color_palette("Set2", imdb.num_classes)
        plt.figure(1, figsize=(15,10))
        plt.clf()
        plt.imshow(im.astype(np.uint8))
        plt.gca().axis('off')

        # skip j = 0, because it's the background class
        n_det = 0
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)            
            keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=True)  
            cls_dets = cls_dets[keep, :]
            n_det += len(inds)
            self.vis_detections(imdb.classes[j], cls_dets, clrs[j])
            
        plt.title('%d objects are detected.' % n_det)
        plt.gca().legend()
        plt.savefig(filename)


    def vis_detections(self, class_name, dets, clr, thresh=0.3):
        """Visual debugging of detections."""                    
        axe = plt.gca()
        for i in xrange(np.minimum(10, dets.shape[0])):
            bbox = dets[i, :4]
            score = dets[i, -1]

            if score > thresh:
                axe.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=clr, linewidth=2.5, label=class_name)                    
                    )
                axe.text(bbox[0], bbox[1]-2, '{:.3f}'.format(score), 
                    bbox=dict(facecolor=clr, alpha=0.5), fontsize=14, color='w')                        

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.pkl')
                
        # caffemodel
        # net.save(str(filename))
        # print 'Wrote snapshot to: {:s}'.format(filename)

        # Save .caffemodel & .solverstate        
        self.solver.snapshot()

        # save original unnormalized params
        bbox_pred = {'0': orig_0, '1': orig_1}        
        with open(filename, 'wb') as fid:
            cPickle.dump(bbox_pred, fid, cPickle.HIGHEST_PROTOCOL)        
                
        filename = filename.replace('.pkl', '.png')

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                filename = self.snapshot()
                model_paths.append(filename)
                # Visualize!                
                if len(self.solver.test_nets) > 0:
                    self.visualize(self.solver.test_nets[0], filename)
                else:
                    self.visualize(self.solver.net, filename)
                
        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb, isTrain):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_AUGMENTATION.FLIP and isTrain:        
        print 'Appending horizontally-flipped training examples...',
        imdb.append_flipped_images()
        print 'done'
    if cfg.TRAIN.USE_AUGMENTATION.CROP and isTrain:
        print 'Appending cropped & resized training examples...',
        imdb.append_crop_resize_images()
        print 'done'
    if cfg.TRAIN.USE_AUGMENTATION.GAMMA and isTrain:
        print 'Appending photometrc transformed training examples...',
        imdb.append_photometric_transformed_images()        
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb_train, roidb_val, imdb_train, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb_train = filter_roidb(roidb_train)
    roidb_val = filter_roidb(roidb_val)

    sw = SolverWrapper(solver_prototxt, roidb_train, roidb_val, imdb_train, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
