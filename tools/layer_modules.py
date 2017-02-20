import os
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

class BaseNets:
	def __init__(self):
		self.layer_param = {'lr_mult': 1, 'decay_mult': 0.005 }

	def AlexNet(self, num_classes):
		datalayer_params={'module':'roi_data_layer.layer', \
						  'layer':'RoIDataLayer', \
						  'param_str': "'num_classes': %d" % num_classes}

		conv_w_filler = {'type': 'gaussian', 'std': 0.01}
		conv_b_filler = {'type': 'constant', 'value': 0.0}

		n = caffe.NetSpect()

		n.data, n.im_info, n.gt_boxes = self.Python(datalayer_params, name='input-data', phase='train')

		##### AlexNet
		# Conv1
		n.conv1 = self.Conv(n.data, 'conv1', n_out=96, k=11, p=0, s=4, g=0, 
			w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])
		n.conv1 = self.ReLU(n.conv1, 'relu1')
		n.norm1 = self.LRN(n.conv1, 'norm1')
		n.pool1 = self.Pool(n.norm1, 'pool1', 'MAX', k=3, p=0, s=2)

		# Conv2
		n.conv2 = self.Conv(n.pool1, 'conv2', n_out=256, k=5, p=2, s=1, g=2, 
			w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])			
		n.conv2 = self.ReLU(n.conv2, 'relu2')
		n.norm2 = self.LRN(n.conv2, 'norm2')
		
		# Conv3
		n.conv3 = self.Conv(n.norm2, 'conv3', n_out=384, k=3, p=1, s=1, g=0, 
			w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])
		n.conv3 = self.ReLU(n.conv3, 'relu3')

		# Conv4
		n.conv4 = self.Conv(n.conv3, 'conv4', n_out=384, k=3, p=1, s=1, g=2, 
			w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])		
		n.conv4 = self.ReLU(n.conv4, 'relu4')

		# Conv5
		n.conv5 = self.Conv(n.conv4, 'conv5', n_out=256, k=3, p=1, s=1, g=2, 
			w_filler=conv_w_filler, b_filler=conv_b_filler, lr_mult=[1, 2], decay_mult=[1, 0])		
		n.conv5 = self.ReLU(n.conv5, 'relu5')
		n.pool5 = self.Pool(n.conv5, 'pool5', 'MAX', 3, 0, 2)

		return n


	def Python(self, python_params, name='data', bottom=None, ntop=1, phase=None):
		params = {'name': name, 'ntop': ntop, 'python_params': python_params}

		if phase is not None:
			mode = {'train':0, 'test':1}
			include = {'phase': mode[phase.lower()]}
			params['include'] = include

		if bottom is not None:
			return L.Python(bottom, **params)
		else:
			return L.Python(**params)

	def MVN(self, bottom, name='mvn'):
		return L.MVN(bottom, name=name)

	def Power(self, bottom, name='power', p=1, a=1, b=0):		# (ax+b)^p
		return L.Power(bottom, name=name, power_param={'power':p, 'scale':a, 'shift':b})

	def Concat(self, bottoms, name='concat', axis=1):		# Channel
		return L.Concat(bottoms[0], bottoms[1], name=name, concat_param={'axis':axis})

	def Reshape(self, bottom, name='dropout', dims=None):	# ex. [-1, 1, 55, 75]				
		return L.Reshape( bottom, name=name, reshape_param={'shape':{'dim': dims}} )

	def Dropout(self, bottom, name='dropout', ratio=0.5):
		return L.Dropout( bottom, name=name, dropout_param={'dropout_ratio': ratio})

	def Full(self, bottom, name='full', n_out=96, w_filler=None, b_filler=None, lr_mult=None, decay_mult=None):
		if w_filler is None:
			w_filler = {'type': 'gaussian', 'std': 0.01}
		if b_filler is None:
			b_filler = {'type': 'constant', 'value': 0.01}

		if lr_mult is not None:			
			params = []
			for lr, decay in zip(lr_mult, decay_mult):
				p = self.layer_param.copy()
				p['lr_mult'] = lr
				p['decay_mult'] = decay
				params.append(p)

			return L.InnerProduct( bottom, name=name, param=params, inner_product_param={'num_output': n_out, \
					'weight_filler': w_filler, 'bias_filler': b_filler} )
		else:		
			return L.InnerProduct( bottom, name=name, inner_product_param={'num_output': n_out, \
					'weight_filler': w_filler, 'bias_filler': b_filler} )

	def Conv(self, bottom, name='conv', n_out=96, k=3, p=0, s=1, g=0, w_filler=None, b_filler=None, lr_mult=None, decay_mult=None):
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
				p = self.layer_param.copy()
				p['lr_mult'] = lr
				p['decay_mult'] = decay
				params.append(p)

			return L.Convolution( bottom, name=name, param=params, convolution_param=conv_params)
		else:		
			return L.Convolution( bottom, name=name, convolution_param=conv_params)

	def ReLU(self, bottom, name='relu'):
		return L.ReLU(bottom, name=name)

	def LRN(self, bottom, name='lrn', ls=5, a=0.0001, b=0.75):	
		return L.LRN(bottom, name=name, lrn_param={'local_size':ls, 'alpha':a, 'beta':b})

	def Pool(self, bottom, name='pool', ptype=None, k=3, p=0, s=2):
		poolMethod = {'max':0, 'ave':1, 'stochastic':2}

		if ptype is None:
			ptype = 0	# MAX
		else:
			ptype = poolMethod[ptype.lower()]
		return L.Pooling(bottom, name=name, pooling_param={'pool':ptype, 'kernel_size':k, 'stride': s, 'pad': p})

	def Gradient_dxdy(self, bottom, name='gradient', n_out=2, k=3, p=0, s=1, lr_mult=None, decay_mult=None):
		conv_params={'num_output': n_out, 'kernel_size': k, 'pad': p, 'stride': s}
		
		if lr_mult is not None:			
			params = []
			for lr, decay in zip(lr_mult, decay_mult):
				p = self.layer_param.copy()
				p['lr_mult'] = lr
				p['decay_mult'] = decay
				params.append(p)

			return L.Convolution( bottom, name=name, param=params, convolution_param=conv_params)
		else:		
			return L.Convolution( bottom, name=name, convolution_param=conv_params)

	def Upsample(self, bottom, name='upsample', factor=2, n_out=3, lr_mult=None, decay_mult=None):

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
				p = self.layer_param.copy()
				p['lr_mult'] = lr
				p['decay_mult'] = decay
				params.append(p)

			return L.Deconvolution( bottom, name=name, param=params, convolution_param=conv_params)
		else:		
			return L.Deconvolution( bottom, name=name, convolution_param=conv_params)
		
	def Silence(self, bottom, name='conv'):
		return L.Silence(bottom, name=name, ntop=0)

	def SoftmaxWithLoss(self, bottom, propagate_down=None, loss_weight=1, loss_param={}):
		pass
