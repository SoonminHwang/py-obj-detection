import os.path as osp
import sys
import numpy as np
import numpy.random as npr

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'lib')
if lib_path not in sys.path:
	sys.path.insert(0, lib_path)

from fast_rcnn.config import cfg, cfg_from_file
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv

import matplotlib.pyplot as plt
import seaborn as sns

def drawBox(box, ax, opt={}):
	import matplotlib.patches as patches	
	try:
		hP = ax.add_patch( patches.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], **opt))
	except:
		import pdb
		pdb.set_trace()
	return hP
	
def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return np.zeros((rois.shape[0], 5), dtype=np.float32)        
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

if __name__ == '__main__':
	
	cfg_from_file('experiments/cfgs/faster_rcnn_end2end_kitti.yml')

	# Load dataset	
	from datasets.kitti import kitti
	imdb = kitti('train', '2012')
	roidb = imdb.roidb

	
	im_scale = float(576) / float(375)	  

	# Load anchors
	from rpn.generate_anchors import generate_anchors
	anchors = generate_anchors(scales=np.array(range(1,10)), ratios=[0.5, 1., 1.5, 2., 2.5, 3.])
	anchors = anchors * im_scale

	num_anchors = anchors.shape[0]
	#height, width = (375, 1242)
	height, width = (int(375*im_scale/16), int(1242*im_scale/16))
	feat_stride = 16

	# 1. Generate proposals from bbox deltas and shifted anchors
	shift_x = np.arange(0, width) * feat_stride
	shift_y = np.arange(0, height) * feat_stride
	shift_x, shift_y = np.meshgrid(shift_x, shift_y)
	shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors        
	A = num_anchors
	K = shifts.shape[0]
	all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
	all_anchors = all_anchors.reshape((K * A, 4))
	total_anchors = int(K * A)

    # only keep anchors inside the image
	inds_inside = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] < 1242) &  # width
        (all_anchors[:, 3] < 375)    # height
	)[0]

	# keep only inside anchors
	#anchors = all_anchors[inds_inside, :]
	anchors = all_anchors

	np.set_printoptions(precision=3)

	
	for i in xrange(len(imdb.image_index)):

		# Load gt boxes
		gt_inds = np.where(roidb[i]['gt_classes'] >= 0)[0]
		gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
		gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :] * im_scale
		gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
	
		# label: 1 is positive, 0 is negative, -1 is dont care
		#labels = np.empty((len(inds_inside), ), dtype=np.float32)
		labels = np.empty((total_anchors, ), dtype=np.float32)
		labels.fill(-1)

		# Computer overlap
		overlaps = bbox_overlaps(
	            np.ascontiguousarray(anchors, dtype=np.float),
	            np.ascontiguousarray(gt_boxes, dtype=np.float))

		argmax_overlaps = overlaps.argmax(axis=1)               # gt index
		#max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]   
		max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]   
		gt_argmax_overlaps = overlaps.argmax(axis=0)            # anchor index
		gt_max_overlaps = overlaps[gt_argmax_overlaps,
	                               np.arange(overlaps.shape[1])]    
		gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]


	    # bg label: assign bg labels first so that positive labels can clobber them
		labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

		# fg label: for each gt, anchor with highest overlap
		labels[gt_argmax_overlaps] = 1

	    # fg label: above threshold IOU (Multiple matching)
		labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

	    # subsample positive labels if we have too many
		num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
		fg_inds = np.where(labels == 1)[0]
		if len(fg_inds) > num_fg:
			disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
			labels[disable_inds] = -1

	    # subsample negative labels if we have too many
		num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
		bg_inds = np.where(labels == 0)[0]
		if len(bg_inds) > num_bg:
			disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
			labels[disable_inds] = -1

		#bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)    
		bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)    
		bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4]).astype(np.float32, copy=False)

 		# map up to original set of anchors
		#labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
		#bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
		#overlaps_ = _unmap(overlaps, total_anchors, inds_inside, fill=0)


		# Reshape to 1 x 1 x (A x H) x W
		#labels_reshape = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
		#labels_reshape = labels_reshape.reshape((1, 1, A * height, width))

		#bbox_targets_reshape = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)


		#####
		idx = np.where( labels == 1 )
		delta = bbox_targets[idx,:][0]
		box = all_anchors[idx, :][0]

		bbox_transform_inv(box, delta)			
		gt_boxes.astype(np.uint16)

		#####
		# order: anchors -> shift (width-> height)
		ii = idx[0]
		#aIdx = ii / K
		#sIdx = ii % K

		aIdx = ii % A
		pos = ii / A

		anchors_orig = generate_anchors(scales=np.array(range(1,10)), ratios=[0.5, 1., 1.5, 2., 2.5, 3.])

		#idx = idx[0] + 1
		all_anchors[idx,:] - shifts[pos,:]
		anchors_orig[aIdx,:]

		yy = pos / width
		xx = pos % width

		import cv2
		img = cv2.imread( imdb.image_path_at(i) )[:,:,(2,1,0)]
		img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

		plt.figure()		
		plt.ion()
		ax = plt.gca()
		ax.imshow(img)

		gtOpt = {'alpha':0.3, 'facecolor':'red'}
		acOpt = {'fill':False, 'edgecolor':'blue', 'linewidth':3}

		drawBox(gt_boxes[0], ax, gtOpt)
		hp = drawBox(all_anchors[idx][0], ax, acOpt)

		hps = [ drawBox(box, ax, acOpt) for box in all_anchors[idx] ]

		plt.show()

		import pdb
		pdb.set_trace()

		ov = bbox_overlaps( np.ascontiguousarray(all_anchors[_idx], dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))

		for ii in range(-A, A, 2):
			_idx = idx[0] + ii + A * width * 2
			hp = drawBox(all_anchors[_idx][0], ax, acOpt)
			ax.set_title('overlaps: {}'.format(overlaps_[_idx][0]))
			#raw_input()
			plt.pause(0.1)
			hp.remove()

		

		###

		# order: anchor -> width -> height

		x = (gt_boxes[:,0] + gt_boxes[:,2]) / 2 / 16
		y = (gt_boxes[:,1] + gt_boxes[:,3]) / 2 / 16

		ii = idx[0]
		xx = int(x)
		yy = int(y)

		aa = ii % A
		pos = ii / A

		r = pos / width
		c = pos % width

		###

		import pdb
		pdb.set_trace()

		tmp = bbox_targets_reshape.reshape(4, A, height, width)


		r4, c1 = np.where( labels_reshape == 1 )
		r1 = np.floor(r4 / float(A)).astype(np.uint16)
		a1 = np.mod(r4, A)



	

		#np.vstack( (bbox_targets[0][a1, r1, c1], bbox_targets[0][a1+1, r1, c1], bbox_targets[0][a1+2, r1, c1], bbox_targets[0][a1+3, r1, c1]) )

		labels_reshape[0][0][r4, c1]


		import pdb
		pdb.set_trace()


	for i in xrange(len(imdb.image_index)):
		roidb[i]['image'] = imdb.image_path_at(i)

		gt_overlaps = roidb[i]['gt_overlaps'].toarray()
	    # max overlap with gt over classes (columns)
		max_overlaps = gt_overlaps.max(axis=1)            
	    # gt class that had the max overlap
		max_classes = gt_overlaps.argmax(axis=1)
	    #roidb[i]['max_classes'] = max_classes
	    #roidb[i]['max_overlaps'] = max_overlaps

		rois = roidb[i]['boxes']

		roidb[i]['bbox_targets'] = np.hstack( (np.ones(rois.shape[0], 1, dtype=np.float32), 
	    									   np.zeros(rois.shape[0], 4, dtype=np.float32)) )
    