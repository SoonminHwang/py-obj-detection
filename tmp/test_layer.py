import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

import cv2
import numpy as np

def _rescale_box(boxes, im_shape, scale=1.0):

	N = len(boxes)

	ct_x = ( boxes[:, 2] + boxes[:,0] ) / 2
	ct_y = ( boxes[:, 3] + boxes[:,1] ) / 2

	width_half = ( boxes[:, 2] - boxes[:,0] + 1 ) / 2
	height_half = ( boxes[:, 3] - boxes[:,1] + 1 ) / 2
	
	new_x1 = max( 0, ct_x - width_half * scale )
	new_y1 = max( 0, ct_y - height_half * scale )

	new_x2 = min( im_shape[1]-1, ct_x + width_half * scale )
	new_y2 = min( im_shape[0]-1, ct_y + height_half * scale )

	return np.hstack( (new_x1, new_y1, new_x2, new_y2) )

if __name__ == '__main__':

	net = caffe.Net('test_layer.pt', caffe.TEST)

	##
	img_input = cv2.imread( 'test_img.png' )
	img = img_input.copy()
	img = img.transpose((2,0,1))
	img = img[np.newaxis, :,:,:]
	input_blob = {'image': img.astype(np.float32, copy=False)}

	# roi = np.array( [0, 200, 200, 300, 300, 0, 150, 150, 350, 350], dtype=np.float32 )
	roi = np.array( [0, 200, 200, 300, 300, 0, 0, 0, 0, 0], dtype=np.float32 )	
	roi = roi.reshape((2,1,5))
	roi[1, 0, 1:] = _rescale_box(roi[0, :, 1:], img_input.shape, 0.8)

	input_blob['roi'] = roi
	# layer_ind = list(net._layer_names).index('roi')
	# net.layers[layer_ind].blobs[0].data[...] = roi

	blobs_out = net.forward(**input_blob)


	# import ipdb
	# ipdb.set_trace()

	import matplotlib.pyplot as plt


	# Upsample test
	out_img = blobs_out['upsample2x'][0].copy()
	out_img = out_img.transpose((1,2,0)).astype(np.uint8)
	cv2.imwrite('output.jpg', out_img)

	# RoiPooling test
	r = roi[0][0]
	plt.imshow(img_input[:,:,(2,1,0)])
	plt.gca().add_patch( plt.Rectangle((r[1], r[2]), r[3]-r[1], r[4]-r[2], fill=False, edgecolor='r', linewidth=2.5) )

	r = roi[1][0]
	plt.gca().add_patch( plt.Rectangle((r[1], r[2]), r[3]-r[1], r[4]-r[2], fill=False, edgecolor='b', linewidth=2.5) )
	plt.savefig('input.jpg')

	roi_img = blobs_out['roi_pool'][0].copy()
	roi_img = roi_img.transpose((1,2,0)).astype(np.uint8)
	cv2.imwrite('roi_pooled_0.jpg', roi_img)

	roi_img = blobs_out['roi_pool'][1].copy()
	roi_img = roi_img.transpose((1,2,0)).astype(np.uint8)
	cv2.imwrite('roi_pooled_1.jpg', roi_img)



	# print('Input shape: %d, %d, %d' % (*img.shape))
	# print('Output shape: %d, %d, %d' % (*out_img.shape))
