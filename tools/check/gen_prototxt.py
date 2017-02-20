try:
	import caffe
except:
	import os
	import sys
	caffe_path = os.path.join(os.path.dirname(__file__), '..', '..', 'caffe-latest', 'python')
	sys.path.insert(0, caffe_path)

	lib_path = os.path.join(os.path.dirname(__file__), '..')
	sys.path.insert(0, lib_path)

	import caffe

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2	
import numpy as np

layer_param = {'lr_mult': 1, 'decay_mult': 0.005 }

##########################################################################################
def get_prototxt(fileNm):

	# with open(fileNm, 'w') as f:
	# 	layerStr = '{}'.format(eigen_refine())
	# 	print layerStr
	# 	f.write(layerStr)

	# with open(fileNm, 'w') as f:
	# 	layerStr = '{}'.format(eigen_nips14())
	# 	print layerStr
	# 	f.write(layerStr)

	with open(fileNm, 'w') as f:
		layerStr = '{}'.format(generate_net())
		print layerStr
		f.write(layerStr)


##########################################################################################
# Eigen - Refine,
def eigen_refine():
	n = caffe.NetSpec()
	n.image, n.gt = EigenData(2, 'input_train', None, 'train')

	##### AlexNet (Global)
	# Conv1
	n.conv1 = Conv(n.image, 'conv1', 96, 11, 0, 4, 0, None, None, [0, 0], [1, 0])
	n.conv1 = ReLU(n.conv1, 'relu1')
	n.norm1 = LRN(n.conv1, 'norm1')
	n.pool1 = Pool(n.norm1, 'pool1', 'MAX', 3, 0, 2)

	# Conv2
	n.conv2 = Conv(n.pool1, 'conv2', 256, 5, 2, 1, 2, None, None, [0, 0], [1, 0])
	n.conv2 = ReLU(n.conv2, 'relu2')
	n.norm2 = LRN(n.conv2, 'norm2')
	
	# Conv3
	n.conv3 = Conv(n.norm2, 'conv3', 384, 3, 1, 1, 0, None, None, [0, 0], [1, 0])
	n.conv3 = ReLU(n.conv3, 'relu3')

	# Conv4
	n.conv4 = Conv(n.conv3, 'conv4', 384, 3, 1, 1, 2, None, None, [0, 0], [1, 0])
	n.conv4 = ReLU(n.conv4, 'relu4')

	# Conv5
	n.conv5 = Conv(n.conv4, 'conv5', 256, 3, 1, 1, 2, None, None, [0, 0], [1, 0])
	n.conv5 = ReLU(n.conv5, 'relu5')
	n.pool5 = Pool(n.conv5, 'pool5', 'MAX', 3, 0, 2)

	##### Main
	w_filler = {'type': 'xavier', 'std': 0.005}
	b_filler = {'type': 'constant', 'value': 0}

	n.fc_main = Full(n.pool5, 'fc_main', 1024, w_filler, b_filler, [0, 0], [1, 0])
	n.fc_main = ReLU(n.fc_main, 'relu6')
	n.fc_main = Dropout(n.fc_main, 'drop6', 0.5)


	w_filler = {'type': 'gaussian', 'std': 0.001}
	b_filler = {'type': 'constant', 'value': 0.5}

	n.fc_depth = Full(n.fc_main, 'fc_depth', 999, w_filler, b_filler, [0, 0], [1, 0])
	n.depth = Reshape(n.fc_depth, 'fc_depth_reshape', [0, 1, 27, 37])
	n.depthMVN = MVN(n.depth, 'mvnDepth_global')

	n.depthMVN2x = Upsample(n.depthMVN, 'g/upsample', 2, 1, [0.0], [0.0])
	
	##### Refine
	w_filler = {'type': 'xavier'}
	b_filler = {'type': 'constant', 'value': 0.01}

	n.r_conv1 = Conv(n.image, 'r/conv1', 96, 11, 2, 2, 0, w_filler, b_filler, [0.001, 0.001], [1, 0])
	n.r_conv1 = ReLU(n.r_conv1, 'r/relu1')
	n.r_norm1 = LRN(n.r_conv1, 'r/norm1')
	n.r_pool1 = Pool(n.r_norm1, 'r/pool1', 'MAX', 2, 1, 2)

	n.r_concat = Concat([n.r_pool1, n.depthMVN2x], 'r/concat', 1)

	# Conv2
	n.r_conv2 = Conv(n.r_concat, 'r/conv2', 64, 5, 2, 1, 1, w_filler, b_filler, [1, 1], [1, 0])
	n.r_conv2 = ReLU(n.r_conv2, 'r/relu2')
	
	# Conv3
	n.r_conv3 = Conv(n.r_conv2, 'r/conv3', 64, 5, 2, 1, 0, w_filler, b_filler, [1, 1], [1, 0])
	n.r_conv3 = ReLU(n.r_conv3, 'r/relu3')

	# Conv4
	n.r_conv4 = Conv(n.r_conv3, 'r/conv4', 64, 5, 2, 1, 2, w_filler, b_filler, [1, 1], [1, 0])
	n.r_conv4 = ReLU(n.r_conv4, 'r/relu4')

	# Conv5
	n.r_depth = Conv(n.r_conv4, 'r/conv5', 1, 3, 1, 1, 1, w_filler, b_filler, [1, 1], [1, 0])
	n.r_depth_p = Power(n.r_depth, 'r/depth_power', 1, 0.01, 0)

	##### Loss
	n.r_depthMVN = MVN(n.r_depth_p, 'r/mvnDepth')
	n.gtMVN = MVN(n.gt, 'mvnGT')

	n.lossMVNDepth = L.EuclideanLoss(n.r_depthMVN, n.gtMVN, name='lossMVNDepth', loss_weight=0.5)
	n.lossDepth = L.EuclideanLoss(n.r_depth_p, n.gt, name='lossABSDepth', loss_weight=0.5)

	return n.to_proto()


##########################################################################################
# Eigen - Global,
def eigen_global():
	n = caffe.NetSpec()
	n.image, n.gt = EigenData(2, 'input_train', None, 'train')
	n.image, n.gt = EigenData(2, 'input_test', None, 'test')

	##### AlexNet
	# Conv1
	n.conv1 = Conv(n.image, 'conv1', 96, 11, 0, 4, 0, None, None, [0.02, 0.02], [1, 0])
	n.conv1 = ReLU(n.conv1, 'relu1')
	n.norm1 = LRN(n.conv1, 'norm1')
	n.pool1 = Pool(n.norm1, 'pool1', 'MAX', 3, 0, 2)

	# Conv2
	n.conv2 = Conv(n.pool1, 'conv2', 256, 5, 2, 1, 2, None, None, [0.02, 0.02], [1, 0])
	n.conv2 = ReLU(n.conv2, 'relu2')
	n.norm2 = LRN(n.conv2, 'norm2')
	
	# Conv3
	n.conv3 = Conv(n.norm2, 'conv3', 384, 3, 1, 1, 0, None, None, [0.02, 0.02], [1, 0])
	n.conv3 = ReLU(n.conv3, 'relu3')

	# Conv4
	n.conv4 = Conv(n.conv3, 'conv4', 384, 3, 1, 1, 2, None, None, [0.02, 0.02], [1, 0])
	n.conv4 = ReLU(n.conv4, 'relu4')

	# Conv5
	n.conv5 = Conv(n.conv4, 'conv5', 256, 3, 1, 1, 2, None, None, [0.02, 0.02], [1, 0])
	n.conv5 = ReLU(n.conv5, 'relu5')
	n.pool5 = Pool(n.conv5, 'pool5', 'MAX', 3, 0, 2)

	##### Main
	w_filler = {'type': 'xavier', 'std': 0.005}
	b_filler = {'type': 'constant', 'value': 0}

	n.fc_main = Full(n.pool5, 'fc_main', 1024, w_filler, b_filler, [1, 1], [1, 0])
	n.fc_main = ReLU(n.fc_main, 'relu6')
	n.fc_main = Dropout(n.fc_main, 'drop6', 0.5)


	w_filler = {'type': 'gaussian', 'std': 0.001}
	b_filler = {'type': 'constant', 'value': 0.5}

	n.fc_depth = Full(n.fc_main, 'fc_depth', 999, w_filler, b_filler, [0.2, 0.2], [1, 0])
	n.depth = Reshape(n.fc_depth, 'fc_depth_reshape', [0, 1, 27, 37])

	n.depthMVN = MVN(n.depth, 'mvnDepth')
	n.gtMVN = MVN(n.gt, 'mvnGT')

	n.lossMVNDepth = L.EuclideanLoss(n.depthMVN, n.gtMVN, name='lossMVNDepth', loss_weight=1)

	return n.to_proto()


##########################################################################################
# Eigen, NIPS '14
def eigen_nips14():

	n = caffe.NetSpec()
	n.image, n.depth = EigenData(2, 'InputData')


	###########################################################
	# Scale 1
	###########################################################

	##### AlexNet (or Caffe ref. Net)
	# Conv1
	n.conv1 = Conv(n.image, 'conv1', 96, 11, 0, 4, 0)
	n.conv1 = ReLU(n.conv1, 'relu1')
	n.norm1 = LRN(n.conv1, 'norm1')
	n.pool1 = Pool(n.norm1, 'pool1', 'MAX', 3, 0, 2)

	# Conv2
	n.conv2 = Conv(n.pool1, 'conv2', 256, 5, 2, 1, 2)
	n.conv2 = ReLU(n.conv2, 'relu2')
	n.norm2 = LRN(n.conv2, 'norm2')
	n.pool2 = Pool(n.norm2, 'pool2', 'MAX', 3, 0, 2)

	# Conv3
	n.conv3 = Conv(n.pool2, 'conv3', 384, 3, 1, 1, 0)
	n.conv3 = ReLU(n.conv3, 'relu3')
	
	# Conv4
	n.conv4 = Conv(n.conv3, 'conv4', 384, 3, 1, 1, 2)
	n.conv4 = ReLU(n.conv4, 'relu4')

	# Conv5
	#n.conv5 = Conv(n.conv4, 'conv5', 256, 5, 2, 1, 2)
	n.conv5 = Conv(n.conv4, 'conv5', 256, 3, 1, 1, 2)
	n.conv5 = ReLU(n.conv5, 'relu5')	
	n.pool5 = Pool(n.conv5, 'pool5', 'MAX', 3, 0, 2)
	#####

	n.scale1_full1 = Full(n.pool5, 'scale1/full1', 4096)
	n.scale1_full1 = ReLU(n.scale1_full1, 'relu_full1')
	n.scale1_full1 = Dropout(n.scale1_full1, 'drop_full1', 0.5)

	n.scale1_full2 = Full(n.scale1_full1, 'scale1/full2', 4070)
	n.scale1_full2_reshape = Reshape(n.scale1_full2, 'scale1/full2_reshape', [-1, 1, 55, 74])


	###########################################################
	# Scale 2
	###########################################################
	# Scale2/Conv1
	n.scale2_conv1 = Conv(n.image, 'scale2/conv1', 64, 9, 0, 2, 0)
	n.scale2_conv1 = ReLU(n.scale2_conv1, 'scale2/relu1')	
	n.scale2_pool1 = Pool(n.scale2_conv1, 'scale2/pool1', 'MAX', 3, 0, 2)
	# concatenate scale2/pool1 + scale1_full2_reshape
	n.scale2_concat = Concat([n.scale2_pool1, n.scale1_full2_reshape], 'scale2/concat', 1)


	# Scale2/Conv2
	n.scale2_conv2 = Conv(n.scale2_concat, 'scale2/conv2', 64, 5, 2, 1, 0)
	n.scale2_conv2 = ReLU(n.scale2_conv2, 'scale2/relu2')	
	
	# Upsample
	n.pred_depth = Upsample(n.scale2_conv2, 'scale2/up2x', 2, 1, 0.0, 0.0)

	
	###########################################################
	# Loss
	###########################################################
	# Gradient (dxdy)
	# n.pred_dxdy = Gradient_dxdy(n.pred_depth, 'pred/dxdy')
	# n.gt_dxdy = Gradient_dxdy(n.depth, 'gt/dxdy')
	# losses = EigenLoss([n.pred_depth, n.depth, n.pred_dxdy, n.gt_dxdy], 'loss', 3, [1, 1, 1])
	# n.lossSqrSum, n.lossSumSqr, n.lossSmooth = losses
	losses = EigenLoss([n.pred_depth, n.depth], 'loss', 3, [1, 1, 1])
	n.lossSqrSum, n.lossSumSqr, n.lossSmooth = losses

	return n.to_proto()

##########################################################################################
# Demo
def generate_net():

	n = caffe.NetSpec()

	n.image, n.depth = EigenData(2)
	n.depth_silence = Silence(n.depth, 'silence_depth')

	n.dxdy = Gradient_dxdy(n.image, 'gradient')
	n.dxdy_silence = Silence(n.dxdy, 'silence_dxdy')

	n.image2x = Upsample(n.image, 'upsample2x', 2)
	n.image2x_silence = Silence(n.image2x, 'silence_image2x')

	return n.to_proto()

##########################################################################################

def EigenData(ntop, name='data', python_params=None, phase=None):
	if python_params is None:
		python_params={'module':'layers.data_layer', 'layer':'EigenDataLayer'}
	
	if phase is not None:
		mode = {'train':0, 'test':1}
		include = {'phase': mode[phase.lower()]}
		return L.Python(name=name, ntop=ntop, include=include, python_param=python_params)
	else:
		return L.Python(name=name, ntop=ntop, python_param=python_params)


def EigenLoss(bottoms, name='loss', ntop=3, loss_weight=[1,1,1], python_params=None):
	if python_params is None:
		python_params={'module':'layers.loss_layer', 'layer':'EigenLossLayer'}

	# return L.Python(bottoms[0], bottoms[1], bottoms[2], bottoms[3], \
	return L.Python(bottoms[0], bottoms[1], \
		name=name, ntop=ntop, loss_weight=loss_weight, python_param=python_params)


def MVN(bottom, name='mvn'):
	return L.MVN(bottom, name=name)

def Power(bottom, name='power', p=1, a=1, b=0):		# (ax+b)^p
	return L.Power(bottom, name=name, power_param={'power':p, 'scale':a, 'shift':b})


def Concat(bottoms, name='concat', axis=1):		# Channel
	return L.Concat(bottoms[0], bottoms[1], name=name, concat_param={'axis':axis})


def Reshape(bottom, name='dropout', s=None):
	if s is None:
		s = [-1, 1, 55, 74]
	
	return L.Reshape( bottom, name=name, reshape_param={'shape':{'dim': s}} )


def Dropout(bottom, name='dropout', ratio=0.5):
	return L.Dropout( bottom, name=name, dropout_param={'dropout_ratio': ratio})


def Full(bottom, name='full', n_out=96, w_filler=None, b_filler=None, lr_mult=None, decay_mult=None):
	if w_filler is None:
		w_filler = {'type': 'gaussian', 'std': 0.01}
	if b_filler is None:
		b_filler = {'type': 'constant', 'value': 0.01}

	if lr_mult is not None:			
		params = []
		for lr, decay in zip(lr_mult, decay_mult):
			p = layer_param.copy()
			p['lr_mult'] = lr
			p['decay_mult'] = decay
			params.append(p)

		return L.InnerProduct( bottom, name=name, param=params, inner_product_param={'num_output': n_out, \
				'weight_filler': w_filler, 'bias_filler': b_filler} )
	else:		
		return L.InnerProduct( bottom, name=name, inner_product_param={'num_output': n_out, \
				'weight_filler': w_filler, 'bias_filler': b_filler} )


def Conv(bottom, name='conv', n_out=96, k=3, p=0, s=1, g=0, w_filler=None, b_filler=None, lr_mult=None, decay_mult=None):
	if w_filler is None:
		w_filler = {'type': 'gaussian', 'std': 0.01}
	if b_filler is None:
		b_filler = {'type': 'constant', 'value': 0.1}

	conv_params = {'num_output': n_out, 'kernel_size': k, 'pad': p, 'stride': s, \
				'weight_filler': w_filler, 'bias_filler': b_filler} 
	if g != 0: conv_params['group'] = g

	
	if lr_mult is not None:			
		params = []
		for lr, decay in zip(lr_mult, decay_mult):
			p = layer_param.copy()
			p['lr_mult'] = lr
			p['decay_mult'] = decay
			params.append(p)

		return L.Convolution( bottom, name=name, param=params, convolution_param=conv_params)
	else:		
		return L.Convolution( bottom, name=name, convolution_param=conv_params)


def ReLU(bottom, name='relu'):
	return L.ReLU(bottom, name=name)


def LRN(bottom, name='lrn', ls=5, a=0.0001, b=0.75):	
	return L.LRN(bottom, name=name, lrn_param={'local_size':ls, 'alpha':a, 'beta':b})


def Pool(bottom, name='pool', ptype=None, k=3, p=0, s=2):
	poolMethod = {'max':0, 'ave':1, 'stochastic':2}

	if ptype is None:
		ptype = 0	# MAX
	else:
		ptype = poolMethod[ptype.lower()]
	return L.Pooling(bottom, name=name, pooling_param={'pool':ptype, 'kernel_size':k, 'stride': s, 'pad': p})


def Gradient_dxdy(bottom, name='gradient', n_out=2, k=3, p=0, s=1, lr_mult=None, decay_mult=None):
	conv_params={'num_output': n_out, 'kernel_size': k, 'pad': p, 'stride': s}
	
	if lr_mult is not None:			
		params = []
		for lr, decay in zip(lr_mult, decay_mult):
			p = layer_param.copy()
			p['lr_mult'] = lr
			p['decay_mult'] = decay
			params.append(p)

		return L.Convolution( bottom, name=name, param=params, convolution_param=conv_params)
	else:		
		return L.Convolution( bottom, name=name, convolution_param=conv_params)


def Upsample(bottom, name='upsample', factor=2, n_out=3, lr_mult=None, decay_mult=None):

	def get_kernel_size(f):
		return 2 * f - f % 2

	def get_pad(f):
		return int( np.ceil( (f-1) / 2.) )

	up_k = get_kernel_size(factor)
	up_s = factor
	up_p = get_pad(factor)

	
	conv_params = {'num_output': n_out, 'group': n_out, \
					'kernel_size': up_k, 'pad': up_p, 'stride': up_s, \
					'weight_filler': {'type': 'bilinear'}, 'bias_term': False}

	if lr_mult is not None:			
		params = []
		for lr, decay in zip(lr_mult, decay_mult):
			p = layer_param.copy()
			p['lr_mult'] = lr
			p['decay_mult'] = decay
			params.append(p)

		return L.Deconvolution( bottom, name=name, param=params, convolution_param=conv_params)
	else:		
		return L.Deconvolution( bottom, name=name, convolution_param=conv_params)
	

def Silence(bottom, name='conv'):
	return L.Silence(bottom, name=name, ntop=0)


##########################################################################################
if __name__ == '__main__':
	
	from datasets.nyud import nyud
	imdb = nyud('train', '2012')

	get_prototxt('train_eigen_nips14.pt')

	solver = caffe.SGDSolver('solver_datalayer_test.pt')
	solver.net.copy_from('../data/imagenet_models/AlexNet.v2.caffemodel')

	layer_ind = list(solver.net._layer_names).index('InputData')
	solver.net.layers[layer_ind].set_imdb(imdb)

	solver.solve()