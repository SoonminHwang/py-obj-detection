# --------------------------------------------------------------------
# Make detection bounding boxes for faster-rcnn
# 	Modified by Soonmin Hwang
#
# Original code: NVcaffe/python/caffe/layers/detectnet/clustering.py
# 	https://github.com/NVIDIA/caffe
# --------------------------------------------------------------------

import caffe
from fast_rcnn.config import cfg
import yaml
import numpy as np
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv

class DetectionLayer(caffe.Layer):
	"""
	** Note that this faster-rcnn based framework only supports batch_size == 1

    * converts detection bbox from predictions to list
    * bottom[0] - scores
        [ num_proposals x 1 ]
    * bottom[1] - bbox
        [ num_proposals x (4 x A) ]
    * bottom[2] - proposal rois made by lib/rpn/proposal_layer.py
		[ num_proposals x 5 (batch_ind, x1, y1, x2, y2) ]
	* bottom[3] - image info
		[ 1 x 3 ]
    * top[0] - list of groundtruth bbox
    	[ num_boxes_after_nms x 6 (cls_ind, x1, y1, x2, y2, score) ]
        
    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'detections'
        # gt_bbox_list is a batch_size x num_boxes_after_nms x 5 blob        
        top: 'bbox_list'
        bottom: 'cls_prob'
        bottom: 'bbox_pred'
        bottom: 'rois'
        bottom: 'im_info'
        python_param {
            module: 'eval.detectnet'
            layer: 'DetectionLayer'            
            param_str : "'nms_thres': 0.3"
        }
        include: { phase: TEST }
    }
    """

	def setup(self, bottom, top):		
		layer_params = yaml.load(self.param_str)
		self.nms_thres = layer_params['nms_thres']


	def forward(self, bottom, top):		
		bbox = postprocess(self, bottom[0].data, bottom[1].data, bottom[2].data, bottom[3].data)
		top[0].reshape(*(bbox.shape))
		top[0].data[...] = bbox


    def reshape(self, bottom, top):
    	"""Reshaping happens during the call to forward."""
    	pass

	def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

def postprocess(self, scores, box_deltas, proposal, im_info):
"""This function is from lib/fast_rcnn/test.py"""

boxes = proposal[:, 1:5] / im_info[2]

pred_boxes = bbox_transform_inv(boxes, box_deltas)
pred_boxes = clip_boxes(pred_boxes, im_info)


results = np.zeros((0, 6), dtype=np.float32)

for cls_ind in xrange(NET.NUM_CLASSES-1):
	cls_ind += 1 # because we skipped background

	cls_boxes = pred_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    
    # CPU NMS is much faster than GPU NMS when the number of boxes
    # is relative small (e.g., < 10k)
    # TODO(rbg): autotune NMS dispatch
    keep = nms(dets, self.nms_thres, force_cpu=True)
    dets = dets[keep, :]
    results = np.vstack( (results, np.insert(dets, 0, cls_ind, axis=1)) )       

return results
