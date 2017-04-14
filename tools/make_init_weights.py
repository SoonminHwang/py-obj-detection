
import sys
import os.path as osp

caffe_path = osp.join(osp.dirname(__file__), '..', 'caffe-latest', 'python')
if caffe_path not in sys.path: sys.path.insert(0, caffe_path)

lib_path = osp.join(osp.dirname(__file__), '..', 'lib')
if lib_path not in sys.path: sys.path.insert(0, lib_path)

import caffe
import numpy as np

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list

expName = 'VGG16_depth0_pedcyc_only'
# expName_cfg = 'VGG16_depth_pedcyc_only'

cfg_file = osp.join('experiments', 'cfgs', 'faster_rcnn_end2end_kitti_' + expName + '.yml')
cfg_from_file(cfg_file)

prototxt = osp.join('models', 'kitti', expName + '.prototxt')
caffemodel = osp.join('data', 'imagenet_models', 'VGG16.v2.caffemodel')

net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)    


## BaseNet
prototxt = osp.join('models', 'kitti', 'VGG16.prototxt')
caffemodel = osp.join('data', 'imagenet_models', 'VGG16.v2.caffemodel')

_net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)    

try:
	for layerName in net.params.keys():
		if '_d' in layerName:
			_layerName = layerName.rstrip('_d')		
			_w = _net.params[_layerName][0].data.copy()
			w = net.params[layerName][0].data.copy()

			_b = _net.params[_layerName][1].data.copy()
			b = net.params[layerName][1].data.copy()

			if _w.shape != w.shape:
				values = _w.shape + w.shape
				print 'Update weights of layer %s' % layerName
				print '(shapes: %dx%dx%dx%d vs. %dx%dx%dx%d)' % values
				w = _w.mean(axis=1)
				net.params[layerName][0].data[...] = w[:,np.newaxis,:,:]
				net.params[layerName][1].data[...] = _b

except:
	import ipdb
	ipdb.set_trace()

# import ipdb
# ipdb.set_trace()

# conv1_1 = _net.params['conv1_1'][0].data.copy()
# conv1_1_c = net.params['conv1_1_c'][0].data.copy()

# old = conv1_1_c.copy()

# conv1_1_c[:,:-1,:,:] = conv1_1

# net.params['conv1_1_c'][0].data[...] = conv1_1_c


# assert( (old != net.params['conv1_1_c'][0].data).any() )

new_caffemodel = osp.join('data', 'imagenet_models', expName + '.v2.caffemodel') 
print 'Save to file: %s' % new_caffemodel
net.save( new_caffemodel )