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
import cv2
import pdb

from datasets.kitti import kitti
from rpn.generate_anchors import generate_anchors

def drawBox(box, ax, opt={}):
	import matplotlib.patches as patches	
	return ax.add_patch( patches.Rectangle( (box[0], box[1]), box[2]-box[0], box[3]-box[1], **opt))
	


DEBUG_SHOW = False

if __name__ == '__main__':
	
	cfg_from_file('experiments/cfgs/faster_rcnn_end2end_kitti.yml')

	np.set_printoptions(precision=3)

	# Load dataset	
	imdb = kitti('train', '2012')
	roidb = imdb.roidb

	#im_scale = float(576) / float(375)
	im_scale = 1.0
	feat_stride = 16
	height, width = (int(375.*im_scale/feat_stride), int(1242.*im_scale/feat_stride))	# feature map size
	

	# Load anchors
	anchor_setting = 'kitti_scale5_ratio4_imscale1.0'
	scales = np.array(range(1,10,2))	
	ratios = np.asarray([0.5, 1.0, 2., 2.5])	
	anchors = generate_anchors(scales=scales, ratios=ratios)
	
	#anchor_setting = 'kitti-data-driven'
	#anchors = imdb.get_anchors()	

	anchors = anchors * im_scale

	A = anchors.shape[0]

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
	K = shifts.shape[0]
	all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
	all_anchors = all_anchors.reshape((K * A, 4))
	total_anchors = int(K * A)

	anchors = all_anchors

	anchor_hist = np.zeros( (A), dtype=np.uint16 )
	
	try:

		num_images = len(imdb.image_index)
		for i in range(num_images):

			if i % 100 == 0:
				print( '[{}/{}] file: {}'.format(i, num_images, imdb.image_path_at(i)))

			# Load gt boxes
			gt_inds = np.where(roidb[i]['gt_classes'] >= 0)[0]
			gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
			gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :] * im_scale
			gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
		
			# label: 1 is positive, 0 is negative, -1 is dont care		
			labels = np.empty((total_anchors, ), dtype=np.float32)
			labels.fill(-1)

			# Computer overlap
			overlaps = bbox_overlaps(
		            np.ascontiguousarray(anchors, dtype=np.float),
		            np.ascontiguousarray(gt_boxes, dtype=np.float))

			argmax_overlaps = overlaps.argmax(axis=1)               # gt index
			max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]   
			gt_argmax_overlaps = overlaps.argmax(axis=0)            # anchor index
			gt_max_overlaps = overlaps[gt_argmax_overlaps,
		                               np.arange(overlaps.shape[1])]    
			gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

			# GT index
			ac_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[1]

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

			bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)    
			bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4]).astype(np.float32, copy=False)

			bbox_targets_reshape = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

			### Debug,
			#idx = np.where( labels == 1 )
			#delta = bbox_targets[idx,:]
			#box = all_anchors[idx, :]
			#assert( bbox_transform_inv(box, delta) == gt_boxes[argmax_overlaps, :4] )

			pos = np.where( labels == 1 )
			best = np.asarray(gt_argmax_overlaps)
			
			#if len(best) > 1: best = best[0]
			anchor_idx = best % A			
			for _idx in anchor_idx:	anchor_hist[_idx] += 1.0
			
			### DEBUG
			#yy = best / A / width;	xx = (best / A) % width
			#aa = range((anchor_idx*4),(anchor_idx*4+4))			
			#assert( (bbox_targets_reshape[0][aa][:, yy, xx].transpose() == bbox_targets[best]).all() )

			if DEBUG_SHOW:
				img = cv2.imread( imdb.image_path_at(i) )[:,:,(2,1,0)]
				img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

				plt.figure(1)		
				plt.clf()
				plt.ion()
				ax = plt.gca()
				ax.imshow(img)

				gtOpt = {'alpha':0.3, 'facecolor':'red'}
				acOpt = {'fill':False, 'edgecolor':'blue', 'linewidth':3}

				[ drawBox(box, ax, gtOpt) for box in gt_boxes ]			
				acOpt['edgecolor'] = 'blue'
				hps = [ drawBox(box, ax, acOpt) for box in all_anchors[pos] ]
				acOpt['edgecolor'] = 'green'
				hps = hps + [drawBox(box, ax, acOpt) for box in all_anchors[best] ]
				plt.title('# of positive: {}, Best overlap: {}'.format(len(pos[0]), overlaps[best].max(axis=0)))
				plt.show()

				pdb.set_trace()

		try:		
			desc_scales = scales.repeat(len(ratios))
			desc_scales = list(scales) * len(ratios)
			desc_ratios = [ r for r in ratios for _ in range(len(scales)) ]

			desc_str = ['s{},r{}'.format(s, r) for s, r, in zip(desc_scales, desc_ratios)]

			anchor_hist_reshape = anchor_hist.reshape( (len(ratios), len(scales)) )
			np.save('experiments/anchors/{}_anchor_hist_reshape.npy'.format(anchor_setting), anchor_hist_reshape)

		except:
			desc_str = ['anchor #{}'.format(_ii) for _ii in range(A) ]
			np.save('experiments/anchors/{}_anchor_hist.npy'.format(anchor_setting), anchor_hist)

		plt.ion()

		plt.figure(2)
		g = sns.barplot(desc_str, anchor_hist, saturation=.5, capsize=0.8, ax=plt.gca())
		plt.gca().set_xticklabels(desc_str, rotation='vertical')
		plt.title('Assigned examples for anchors (Total)')
		plt.show()
		plt.savefig('experiments/anchors/{}_assigned_anchors.jpg'.format(anchor_setting))

		
		plt.figure(3)
		desc_str = ['scale {}'.format(s) for s in scales]
		g = sns.barplot(desc_str, anchor_hist_reshape.sum(axis=0), saturation=.5, capsize=0.8, ax=plt.gca())
		plt.gca().set_xticklabels(desc_str, rotation='vertical')
		plt.title('Assigned examples for anchors (Scale)')
		plt.savefig('experiments/anchors/{}_assigned_anchors_scale.jpg'.format(anchor_setting))


		plt.figure(4)
		desc_str = ['ratio {}'.format(r) for r in ratios]
		g = sns.barplot(desc_str, anchor_hist_reshape.sum(axis=1), saturation=.5, capsize=0.8, ax=plt.gca())
		plt.gca().set_xticklabels(desc_str, rotation='vertical')
		plt.title('Assigned examples for anchors (Ratio)')
		plt.savefig('experiments/anchors/{}_assigned_anchors_ratio.jpg'.format(anchor_setting))

		plt.ion()
		plt.show()
		
	except:
		pdb.set_trace()

	

	pdb.set_trace()