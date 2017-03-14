import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

import cv2
import numpy as np

net = caffe.Net('test_layer.pt', caffe.TEST)

img = cv2.imread( 'test_img.png' )
img = img.transpose((2,0,1))
img = img[np.newaxis, :,:,:]

input_blob = {'image': img.astype(np.float32, copy=False)}
blobs_out = net.forward(**input_blob)

import matplotlib.pyplot as plt
out_img = blobs_out['upsample2x'][0].copy()
out_img = out_img.transpose((1,2,0)).astype(np.uint8)
plt.imshow(out_img)
plt.savefig('test_img_out.jpg')

import ipdb
ipdb.set_trace()


# print('Input shape: %d, %d, %d' % (*img.shape))
# print('Output shape: %d, %d, %d' % (*out_img.shape))
