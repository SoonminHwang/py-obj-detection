import os.path as osp
import sys
import numpy as np
import numpy.random as npr

import _init_paths
from datasets.kaist import kaist

from fast_rcnn.config import cfg, cfg_from_file
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv

import matplotlib.pyplot as plt
import seaborn as sns

import pdb
from pprint import pprint as pp

if __name__ == '__main__':
	
	imdb = kaist('train01', '2015')
	roidb = imdb.roidb

	boxes = np.vstack( ( r['boxes'][r['gt_classes'] == 1, :] for r in roidb if r['gt_classes'].any() ) )
	boxes_wh = boxes[:,2:] - boxes[:, :2]

	from sklearn.cluster import KMeans
	
	np.set_printoptions(precision=2)
	
	for k in range(5, 12):
		km = KMeans(n_clusters=k)
		km.fit(boxes_wh)

		boxes_wh_k = [boxes_wh[km.labels_==l, :] for l in range(k)]

		stds = [np.mean((ctr - wh)**2, axis=0) for ctr, wh in zip(km.cluster_centers_, boxes_wh_k)]
		nSamples = [len(wh) for wh in boxes_wh_k]

		pp('### k = %d'%(k))
		pp('    nSamples:')
		pp(nSamples)
		pp('    stds:')
		pp(stds)		

	k_ = 9		# Selected # of anchors
	km = KMeans(n_clusters=k_)
	km.fit(boxes_wh)
	wh_centers = np.vstack( (km.cluster_centers_) )

	area = wh_centers[:,0] * wh_centers[:,1]
	idx = area.argsort()
	wh_centers = wh_centers[idx, :]
	anchors = np.hstack( (-1 * wh_centers/2., wh_centers/2.))

	pp('')
	pp('###### Data-driven anchors:')
	pp(anchors)

	plt.figure()
	ax = plt.gca()
	clrs = sns.color_palette("Set2", 10)
	for clr, aa in zip(clrs, anchors):
		ax.add_patch( plt.Rectangle((aa[0], aa[1]), aa[2] - aa[0], aa[3] - aa[1], fill=False, edgecolor=clr, linewidth=3.5) )
	
	ax.set_xlim(-200, 200)
	ax.set_ylim(-200, 200)

	plt.savefig('anchors_kaist.jpg')

	pdb.set_trace()