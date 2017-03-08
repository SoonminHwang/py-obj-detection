import os
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layer_modules import BaseNets

def write_solver(output_dir, base_lr=0.01, momentum=0.9, test_iter=10, test_interval=15, display=5, snapshot=0,
    snapshot_prefix='faster_rcnn', lr_policy='step', stepsize=15000, gamma=0.1, weight_decay=0.0005, 
    net='trainval.prototxt', train_net='train.prototxt', test_net='test.prototxt', 
    max_iter=37500, test_initialization='true', test_compute_loss='true', average_loss=10, iter_size=16):

    sp = {}

    # critical
    sp['base_lr'] = str(base_lr)
    sp['momentum'] = str(momentum)

    # speed
    if test_iter != 0 and test_interval != 0:
    	sp['test_iter'] = str(test_iter)
    	sp['test_interval'] = str(test_interval)

    # looks
    sp['display'] = str(display)
    sp['snapshot'] = str(snapshot)
    sp['snapshot_prefix'] = '"%s"'%snapshot_prefix

    # learning rate policy
    sp['lr_policy'] = '"%s"'%lr_policy
    sp['stepsize'] = str(stepsize)

    # important, but rare
    sp['gamma'] = str(gamma)
    sp['weight_decay'] = str(weight_decay)
    sp['net'] = '"%s"'%net
    # sp['train_net'] = '"%s"'%train_net
    # sp['test_net'] = '"%s"'%test_net

    # pretty much never change these.
    sp['max_iter'] = str(max_iter)
    sp['test_initialization'] = test_initialization
    sp['test_compute_loss'] = test_compute_loss
    sp['average_loss'] = str(average_loss)
    sp['iter_size'] = str(iter_size)
    
    solver_file = os.path.join(output_dir, 'models', 'solver.prototxt')
    with open( solver_file, 'w') as f:
        for key, value in sorted(sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))

    return solver_file

class FasterRCNN(BaseNets):
	def __init__(self, dst_path, num_classes, num_anchors, basenet='AlexNet', rpn_from={'layer':'conv5', 'stride':16}):
		self._dst_path = dst_path
		self._basenet = basenet
		self._num_classes = num_classes
		self._num_anchors = num_anchors
		self._rpn_from_layer = rpn_from['layer']
		self._rpn_stride = rpn_from['stride']


	def WriteNet(self):
		if self._basenet == 'AlexNet':
			n = AlexNet(self._num_classes)

			n.conv5 = self.Conv(n.conv4, 'conv5', n_out=256, k=3, p=1, s=1, g=2, 
			w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])		

			conv_w_filler = {'type': 'gaussian', 'std': 0.01}
			conv_b_filler = {'type': 'constant', 'value': 0.0}
			
			# RPN
			n.rpn_output = self.Conv(n.conv5, 'rpn_conv_3x3', n_out=256, k=3, p=1, s=1, g=0,
				w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])
			n.rpn_output = self.ReLU(n.rpn_output, 'rpn_relu_3x3')

			# RPN-score
			n.rpn_cls_score = self.Conv(n.rpn_output, 'rpn_cls_score', 
				n_out=2*self._num_anchors, k=1, p=0, s=1, g=0,		# FC.
				w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])
			n.rpn_cls_score_reshape = self.Reshape(n.rpn_cls_score, 'rpn_cls_score_reshape', dims=[0, 2, -1, 0])
			
			anchor_target_layer_params={'module':'rpn.anchor_target_layer', \
						  'layer':'AnchorTargetLayer', \
						  'param_str': "'feat_stride': %d" % self._rpn_stride}
			anchor_target_layer_bottoms=['rpn_cls_score', 'gt_boxes', 'in_info', 'data']
			n.rpn_labels, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights = \
				self.Python(anchor_target_layer_params, name='rpn-data', bottom=anchor_target_layer_bottoms)


			n.rpn_bbox_pred = self.Conv(n.rpn_output, 'rpn_bbox_pred', 
				n_out=4*self._num_anchors, k=1, p=0, s=1, g=0,		# FC.
				w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])

			n.rpn_output = self.ReLU(n.rpn_output, 'rpn_relu_3x3')

		




