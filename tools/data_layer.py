import os
import sys
caffe_path = os.path.join(os.path.dirname(__file__), '..', '..', 'caffe-latest', 'python')
sys.path.insert(0, caffe_path)

lib_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, lib_path)

import caffe
from config import cfg, cfg_from_list
import numpy as np
import yaml
from multiprocessing import Process, Queue

from datasets.nyud import nyud
import pdb

DEBUG = False

class EigenDataLayer(caffe.Layer):
    
    # def set_imdb(self, imdb):
    #     self.imdb = imdb
    #     self._shuffle_inds()

    def _shuffle_inds(self):
        """Randomly permute the training roidb."""        
        self._perm = np.random.permutation(np.arange(self.imdb.num_images))
        self._cur = 0

    def _get_next_minibatch_inds(self):        
        if self._cur + cfg.TRAIN.BATCH_SIZE >= self.imdb.num_images:
            self._shuffle_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""        
        db_inds = self._get_next_minibatch_inds()
        
        sz = cfg.TRAIN.INPUT_SIZE
        rand_w = np.random.randint(sz[1]*0.8, sz[1], cfg.TRAIN.BATCH_SIZE)
        rand_h = np.random.randint(sz[0]*0.8, sz[0], cfg.TRAIN.BATCH_SIZE)
        rand_off = np.random.randint(0, self.max_off)
        rand_crop = (rand_h, rand_w, rand_off)

        image = np.asarray([self.imdb.get_rgb_image(i, rand_crop) for i in db_inds])
        image = image.transpose((0,3,1,2))    # B x C x H x W

        depth = np.asarray([self.imdb.get_depth_image(i, rand_crop) for i in db_inds])
        depth = depth.transpose((0,3,1,2))    # B x C x H x W
        # depth = depth[:,np.newaxis,1:-1,1:-1]    # B x 1 x H x W

        # import pdb
        # pdb.set_trace()

        if DEBUG:
            # print('phase: %d, file: %s'%(self.phase, self.imdb.file_name_from_index(db_inds[0])))
            
            import matplotlib.pyplot as plt
            fig = plt.figure(self.phase+11)
            axes = [ fig.add_subplot(2,1,n+1) for n in range(2) ]            
            img = ( image[0].transpose((1,2,0)) * 255.0 ).astype(np.uint8)
            dep = depth[0][0]

            axes[0].imshow(img)
            axes[1].imshow(dep, cmap='gray')
            plt.title('phase: %d, file: %s'%(self.phase, self.imdb.file_name_from_index(db_inds[0])))

            plt.ion()
            plt.pause(1)
            # pdb.set_trace()

            

        blobs = {'image': image, 'gt': depth}

        # if self._name_to_top_map.has_key('roi'):
        #     blobs['roi'] = self.roi

        return blobs

    
    def setup(self, bottom, top):
        """Setup the EigenDataLayer."""
        layer_params = yaml.load(self.param_str)
        dtype = layer_params['data_type']
        year = layer_params['year']        

        self.imdb = nyud(dtype, year)
        self._shuffle_inds()

        self._cur = 0
        self._name_to_top_map = {}
        
        osz = cfg.TRAIN.RAW_INPUT_SIZE
        isz = cfg.TRAIN.IMAGE_SIZE
        sz = cfg.TRAIN.INPUT_SIZE
        tsz = cfg.TRAIN.TARGET_SIZE

        self.max_off = (isz[0] - sz[0])/2

        top[0].reshape(cfg.TRAIN.BATCH_SIZE, 3, sz[0], sz[1])   # image
        self._name_to_top_map['image'] = 0
        
        top[1].reshape(cfg.TRAIN.BATCH_SIZE, 1, tsz[0], tsz[1])   # depth
        self._name_to_top_map['gt'] = 1

        # if len(top) == 3:
        #     top[2].reshape(cfg.TRAIN.BATCH_SIZE, 5)
        #     self._name_to_top_map['roi'] = 2

        # box = (2, 2, 57, 76)
        # roi = [(b,) + box for b in range(cfg.TRAIN.BATCH_SIZE)]
        # self.roi = np.asarray( roi )

        print 'Setup EigenDataLayer'        

    def forward(self, bottom, top):        
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()            

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def datalayer_test(imdb, batch_size=4):    

    from caffe import layers as L, params as P, to_proto
    from caffe.proto import caffe_pb2

    w_filler_params = {'weight_filler': {'type': 'xavier'}}
    b_filler_params = {'bias_filler': {'type': 'constant', 'value': 0.01}}
    
    n = caffe.NetSpec()
    n.image, n.depth = L.Python(name='data_train', ntop=2, include={'phase':0}, \
        python_param={'module':'data_layer', 'layer':'EigenDataLayer', 'param_str': "{'data_type': 'train', 'year': '2012'}"})

    n.image, n.depth = L.Python(name='data_test', ntop=2, include={'phase':1}, \
        python_param={'module':'data_layer', 'layer':'EigenDataLayer', 'param_str': "{'data_type': 'test', 'year': '2012'}"})
    
    n.image_s = L.Silence(n.image, name='silence_image', ntop=0)
    n.depth_s = L.Silence(n.depth, name='silence_depth', ntop=0)

    # n.conv1_1 = L.Convolution( n.image, name='conv1_1', convolution_param={'num_output': 64, 'kernel_size': 3, 'pad': 0, 'weight_filler': {'type': 'xavier'}, 'bias_filler': {'type': 'constant', 'value': 0.01} } )
    # n.conv1_1 = L.ReLU( n.conv1_1 )
    
    # n.conv1_2 = L.Convolution( n.conv1_1, name='conv1_2', convolution_param={'num_output': 64, 'kernel_size': 3, 'pad': 0, 'weight_filler': {'type': 'xavier'}, 'bias_filler': {'type': 'constant', 'value': 0.01} } )
    # n.conv1_2 = L.ReLU( n.conv1_1 )
    # n.pool1_2 = L.Pooling


    # n.conv2 = L.Convolution( n.lrn1, name='conv2', convolution_param={'num_output': 64, 'kernel_size': 3, 'pad': 1, 'weight_filler': {'type': 'xavier'}, 'bias_filler': {'type': 'constant', 'value': 0.01} } )    
    # n.conv2 = L.ReLU( n.conv2 )
    # n.lrn2 = L.LRN( n.conv2, name='lrn2', lrn_param={'local_size': 5, 'alpha': 0.0001, 'beta': 0.75} )

    # n.pred = L.Convolution( n.lrn2, name='conv3', convolution_param={'num_output': 1, 'kernel_size': 3, 'pad': 1, 'weight_filler': {'type': 'xavier'}, 'bias_filler': {'type': 'constant', 'value': 0.01} } )    
    
    # n.lossSqrSum, n.lossSumSqr, n.lossSmooth = L.Python(n.pred, n.depth, name='loss', ntop=3, loss_weight=[1,1,1], \
    #             python_param={'module':'loss_layer', 'layer':'EigenLossLayer'})
    # n.loss = L.EuclideanLoss(n.pred, n.depth, ntop=1)



    # n.lrn1_2 = L.LRN( n.conv1_1, name='lrn1', lrn_param={'local_size': 5, 'alpha': 0.0001, 'beta': 0.75} )

    return n.to_proto()

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    # _, axes = plt.subplots(2,2,figsize=(10,15))
    # axes = axes.flatten()

    # from datasets.nyud import nyud
    # imdb = nyud('train', '2012')

    # with open('datalayer_test.pt', 'w') as f:
    #     layerStr = '{}'.format(datalayer_test(imdb))
    #     print layerStr
    #     f.write(layerStr)

    cfg_from_list(['TRAIN.BATCH_SIZE', '1'])

    caffe.set_mode_gpu()
    caffe.set_device(0)

    # net = caffe.Net('datalayer_test.pt', caffe.TRAIN)
    solver = caffe.SGDSolver('solver.pt')

    # layer_ind = list(solver.net._layer_names).index('data')
    # solver.net.layers[layer_ind].set_imdb(imdb)

    # conv_filters = solver.net.layers[1].blobs[0].data
    # print conv_filters
    print 'Start training..'

    # plt.ion()

    for ii in range(400):
        # out = net.forward()
        solver.step(1)
                
        # import ipdb
        # ipdb.set_trace()

        # if ii % 1 == 0:

        # layer_ind = list(solver.net._layer_names).index('conv2')
        # conv_filters = solver.net.layers[layer_ind].blobs[0].data
        
        # pred = solver.net.blobs['pred'].data[0][0]
        # pred += 1.0
        # pred *= 0.5*255.0

        # img = solver.net.blobs['image'].data[0].transpose((1,2,0))
        # img += 1.0
        # img *= 255.0

        # target = solver.net.blobs['depth'].data[0][0]
        # target += 1.0
        # target *= 0.5*255.0

        # print conv_filters

        # layer_ind = list(solver.net._layer_names).index('data')
        # print('Phase: %d' % solver.net.layers[layer_ind].phase)

        # axes[0].imshow( conv_filters[0].transpose((1,2,0)) )
        # axes[1].imshow( pred, cmap='gray' )
        # axes[2].imshow( img.astype(np.uint8) )
        # axes[3].imshow( target, cmap='gray' )
        # plt.pause(1)

        pdb.set_trace()
        # image = out['result'].copy()
        # depth = out['depth'].copy()

        # axes[0].imshow( image[0].transpose((1,2,0)).astype(np.uint8) )
        # axes[1].imshow( depth[0][0] )


        # plt.title('conv_filter, iter {}'.format(ii))
        # plt.savefig('decolor-eigenloss/predict_{:02d}.jpg'.format(ii))