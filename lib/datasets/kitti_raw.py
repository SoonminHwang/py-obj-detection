# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# This file is tested only for end2end with RPN mode.
# --------------------------------------------------------
# If you add another dataset,
#   please modify follow files.
#       - json instances (converted raw annotation file)
#       - this file
#       - roi_data_layer/minibatch.py (input layer)
#       - rpn/anchor_target_layer.py (generate GT for RPN)
#       - rpn/proposal_layer.py (produce RoIs in pixel: sort, nms)
#       - rpn/proposal_target_layer.py (generate GT for RCNN)
# --------------------------------------------------------

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from fast_rcnn.config import cfg, cfg_from_file
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import json
import uuid
# COCO API
from pycocotools.kitti import KITTI
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

import matplotlib.pyplot as plt    
import seaborn as sns
import ipdb
# from rpn.generate_anchors import generate_anchors

class kitti(imdb):    

    def __init__(self, image_set, year):
        imdb.__init__(self, 'kitti_' + year + '_' + image_set)
        # KITTI specific config options
        self.config = {'cleanup' : True,                       
                       'hRng' : [20, np.inf], # Min. 20 x 50 or 25 x 40
                       'occLevel' : [0, 1, 2],       # 0: fully visible, 1: partly occ, 2: largely occ, 3: unknown
                       'truncRng' : [0, 0.5]     # Only partially-truncated
                      }

        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'kitti')

        with open( os.path.join(self._data_path, 'ImageSets_3DOP', image_set + '.txt'), 'r' ) as f:
            self._image_index = [ int(num) for num in f.readlines() ] 
                
        # load KITTI API, classes, class <-> id mappings
        # self._KITTI = KITTI(self._get_ann_file())

        # Below classes are only used for training.  
        # In training set, 
        # ['Van', 'Truck', 'Person_sitting'] classes are marked as 
        #   ['Car', Car', 'Pedestrian'] respectively for convenience      
        
        categories = ['Pedestrian', 'Cyclist', 'Car', 'Ignore']       
        self._cat_maps = {  'Pedestrian': categories[0],
                            'Person_sitting': categories[0],
                              'Cyclist': categories[1],
                              'Car': categories[2],         
                              'Van': categories[2],         
                              'Truck': categories[-1],       # Exclude!
                              'Tram': categories[-1],
                              'Misc': categories[-1],
                              'DontCare': categories[-1] }

        # self._cat_maps = {  'Pedestrian': categories[0],
        #                     'Person_sitting': categories[0],
        #                       'Cyclist': categories[1],
        #                       'Car': categories[2],
        #                       'Van': categories[2],
        #                       'Truck': categories[3],       # Exclude!
        #                       'Tram': categories[3],
        #                       'Misc': categories[3],
        #                       'DontCare': categories[3] }

        self._classes = tuple(['__background__'] + categories)
        # self._raw_cat_ids = self._KITTI.getCatIds(catNms=categories)

        # cats = self._KITTI.loadCats(self._raw_cat_ids)
        # self._classes = tuple(['__background__'] + [c['name'] for c in cats])
                
        self._class_to_ind = dict(zip(self.classes, xrange(len(self._classes))))        
        # self._class_to_kitti_cat_id = dict(zip([c['name'] for c in cats], self._raw_cat_ids))        

        # self._image_index = self._load_image_set_index()
        
        # Default to roidb handler
        assert cfg.TRAIN.PROPOSAL_METHOD == 'gt', \
            'Only supports "gt" for proposal method for kitti dataset.'
        
        self.set_proposal_method('gt')
        #self.competition_mode(False)

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.

        # For KITTI dataset, raw-train set provided by the original author is divided into train/val set.
        #   So, we call raw-train set trainval2012 consisting of train2012 and val2012.        
        self._view_map = {            
            'val' : 'training',
            'train' : 'training',
            'trainval' : 'training',
            'test' : 'testing'
        }
        
        # E.g. train2012/val2012 -> self._data_name = 'trainval2012'
        #      test2012 -> self._data_name = 'test2012'
        # kitti_name = image_set + year  # e.g., "val2014"
        # self._data_name = (self._view_map[kitti_name]
        #                    if self._view_map.has_key(kitti_name)
        #                    else kitti_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        #self._gt_splits = ['train', 'val', 'minival']

        # Add input modalities        
        # self.input_types = ['image', 'depth']
        self.input_types = cfg.INPUT

        self.input_path_at = []
        self.input_path_from_index = []
        # self.input_preprocess = []

        for modal in self.input_types:
            input_path_at = eval('self.' + modal + '_path_at')
            input_path_from_index = eval('self.' + modal + '_path_from_index')
            # input_preprocess = eval('self.' + modal + '_preprocess')
            
            self.input_path_at.append( input_path_at )
            self.input_path_from_index.append( input_path_from_index )


    # def image_preprocess(self, im, pixel_means, target_size, max_size):
    #     """Mean subtract and scale an image for use in a blob."""
    #     im = im.astype(np.float32, copy=False)
    #     im -= pixel_means
    #     im_shape = im.shape
    #     im_size_min = np.min(im_shape[0:2])
    #     im_size_max = np.max(im_shape[0:2])
    #     im_scale = float(target_size) / float(im_size_min)
    #     # Prevent the biggest axis from being more than MAX_SIZE
    #     if np.round(im_scale * im_size_max) > max_size:
    #         im_scale = float(max_size) / float(im_size_max)
    #     im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
    #                     interpolation=cv2.INTER_LINEAR)

    #     return im, im_scale

    # def depth_preprocess(self, im, pixel_means, target_size, max_size):
    #     """Mean subtract and scale an image for use in a blob."""
    #     im = im.astype(np.float32, copy=False)
    #     im -= pixel_means
    #     im_shape = im.shape
    #     im_size_min = np.min(im_shape[0:2])
    #     im_size_max = np.max(im_shape[0:2])
    #     im_scale = float(target_size) / float(im_size_min)
    #     # Prevent the biggest axis from being more than MAX_SIZE
    #     if np.round(im_scale * im_size_max) > max_size:
    #         im_scale = float(max_size) / float(im_size_max)
    #     im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
    #                     interpolation=cv2.INTER_LINEAR)

    #     return im, im_scale

    # def _get_ann_file(self):
    #     prefix = 'instances' if self._image_set.find('test') == -1 \
    #                          else 'image_info'            
        
    #     return osp.join(self._data_path, 'annotations',
    #                     prefix + '_' + self._image_set + self._year + '.json')        

    # def _load_image_set_index(self):
    #     """
    #     Load image ids.
    #     """
    #     # image_ids = self._KITTI.getImgIds()
    #     # return image_ids
    #     return self._subset_index

    # def _get_widths(self):
    #     anns = self._KITTI.loadImgs(self._image_index)
    #     widths = [ann['width'] for ann in anns]
    #     return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_file = osp.join(self._data_path, self._view_map[self._image_set], 'image_2', '%06d.png' % index)
        # im_ann = self._KITTI.loadImgs(index)[0]
        # fName = self._image_index[index]
        # image_path = osp.join(self._data_path, 'images', self._data_name, im_ann['file_name'])
        
        assert osp.exists(image_file), \
                'Path does not exist: {}'.format(image_file)
        return image_file

    def depth_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.depth_path_from_index(self._image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # im_ann = self._KITTI.loadImgs(index)[0]                    
        # fileName = im_ann['file_name'].replace('png', 'bin')
        # image_path = osp.join(self._data_path, 'depths', self._data_name, fileName)
        # depth_file = osp.join(self._data_path, self._view_map[self._image_set], 'disparity_2', '%06d.bin' % index)
        depth_file = osp.join(self._data_path, self._view_map[self._image_set], 'velo_dispmap', '%06d.npy' % index)

        assert osp.exists(depth_file), \
                'Path does not exist: {}'.format(depth_file)
        return depth_file

    def label_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.label_path_from_index(self._image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        label_file = osp.join(self._data_path, self._view_map[self._image_set], 'label_2', '%06d.txt' % index)
        # im_ann = self._KITTI.loadImgs(index)[0]
        # fName = self._image_index[index]
        # image_path = osp.join(self._data_path, 'images', self._data_name, im_ann['file_name'])
        
        assert osp.exists(label_file), \
                'Path does not exist: {}'.format(label_file)
        return label_file


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _read_annotation_file(self, label_file):

        anns = []

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = line.split(' ')
                nums = [float(num) for num in d[1:]]
                clsStr = d[0]
                trunc, occ, alpha = nums[:3]
                left, top, right, bottom = nums[3:7]
                height_3d, width_3d, length_3d, x, y, z, rot_y = nums[7:]
                                
                #cat_id  = [ cat['id'] for cat in data['categories'] if clsStr == cat['name'] ][0]
                # cat_id = class_to_ind[clsStr]

                cat_id = self._class_to_ind[ self._cat_maps[clsStr] ]
                        
                bbox    = [left, top, right-left, bottom-top]
                
                ann = {}                
                ann['category_id']  = cat_id                        
                ann['bbox']         = bbox
                ann['trunc']        = trunc
                ann['occ']          = occ
                ann['alpha']        = alpha
                ann['dim_3d']       = [height_3d, width_3d, length_3d]
                ann['loc_3d']       = [x, y, z]
                ann['rot_y']        = rot_y

                anns.append(ann)

        return anns

    def _load_kitti_annotation(self, index):
        """
        Loads KITTI bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        # im_ann = self._KITTI.loadImgs(index)[0]

        # Load annotation file
        label_file = self.label_path_from_index(index)
        objs = self._read_annotation_file(label_file)

        import cv2
        image_file = self.image_path_from_index(index)
        height, width = cv2.imread(image_file).shape[:2]

        # Follow 'demo_load_kitti_dataset.py by Soonmin'        
        # hRng, occLevel, tRng = self.config['hRng'], self.config['occLevel'], self.config['truncRng']

        # Load annotation ids
        # annIds = self._KITTI.getAnnIds(imgIds=index, catIds=self._raw_cat_ids, 
                                       # hRng=hRng, occLevel=occLevel, truncRng=tRng)
        #annIds = self._KITTI.getAnnIds(imgIds=index, hRng=hRng, occLevel=occLevel, truncRng=tRng)
        
        # objs = self._KITTI.loadAnns(annIds)        

        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))            
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            
            cls = obj['category_id']

            ### If obj does not satisfy config condition, cls override to 'Ignore'
            # 1. height range
            hRng = self.config['hRng']
            if obj['bbox'][3] < hRng[0] or obj['bbox'][3] > hRng[1]:
                cls = self._class_to_ind['Ignore']

            # 2. occlusion level
            occLevel = self.config['occLevel']
            if obj['occ'] not in occLevel:
                cls = self._class_to_ind['Ignore']

            # 3. truncation range
            tRng = self.config['truncRng']
            if obj['trunc'] < tRng[0] or obj['trunc'] > tRng[1]:
                cls = self._class_to_ind['Ignore']

            obj['class'] = cls

            # All valid annotations must satisfy below condition
            # if x2 >= x1 and y2 >= y1:
            if x2 >= x1 and y2 >= y1 and cls != self._class_to_ind['Ignore']:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        objs = valid_objs            
        num_objs = len(objs)        

        # In traffic scene datasets (e.g. KITTI, KAIST),
        #   some images may not contain any target object instance.
        #   Then, num_objs == 0.
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Lookup table to map from KITTI category ids to our internal class indices                        
        # kitti_cat_id_to_class_ind = dict([(self._class_to_kitti_cat_id[cls], self._class_to_ind[cls])
                                         # for cls in self._classes[1:]])
                    
        for ix, obj in enumerate(objs):
            # cls = kitti_cat_id_to_class_ind[ obj['category_id'] ]                      
            # gt_classes[ix] = cls
            cls = obj['class']
            boxes[ix, :] = obj['clean_bbox']            
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
                            
        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)        

        return {'width' : width,
                'height' : height,
                'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,      # Data augmentation
                'gamma' : False,        # Data augmentation
                'crop' : None,          # Data augmentation
                'jitter' : False
                }


    def append_ped_cyc_images(self):
        roidb_old = self.roidb[:]

        ped_ind, cyc_ind = self._class_to_ind['Pedestrian'], self._class_to_ind['Cyclist']

        for ix, r in enumerate(roidb_old):
            if ped_ind in r['gt_classes'] or cyc_ind in r['gt_classes']:
                r['gamma'] = True
                self.roidb.append(r)
                self._image_index.append( self._image_index[ix] )
                

    def _kitti_results_template(self):
        # class string, x1 y1 x2 y2, score
        resStr = '{:s} -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.4f}\n'
        return resStr

    def _write_kitti_results_file(self, all_boxes, output_dir):
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        
        for im_ind, index in enumerate(self.image_index):
            im_name = os.path.basename( self.image_path_at(im_ind) )
            im_name = im_name.replace('png', 'txt')
            
            with open(os.path.join(output_dir, im_name), 'w') as f:

                for cls_ind, cls in enumerate(self.classes[:-1]):
                    if cls == '__background__': continue                
                    
                    dts = all_boxes[cls_ind][im_ind].astype(np.float)
                    if dts == []: continue

                    for dt in dts:
                        f.write( 
                            self._kitti_results_template().format(cls, dt[0], dt[1], dt[2], dt[3], dt[4])
                        )

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_kitti_results_file(all_boxes, output_dir)
        self._do_python_eval(output_dir)


    def _do_python_eval(self, result_dir):
        from kitti_eval import EvalKITTI
        # gtDir = os.path.join( cfg.ROOT_DIR, 'data', 'kitti', 'annotations', self._image_set + self._year )
        gtDir = os.path.join( cfg.ROOT_DIR, 'data', 'kitti', 'training', 'label_2' )
        
        if os.path.exists(gtDir):
            # validation set
            eval_kitti = EvalKITTI(gtDir, result_dir, basePth='')
            eval_kitti.evaluate()
        else:
            # test set
            print '"%s" does not exist. Cannot evaluate detection results.'

   
# For Debugging purpose,   
def get_assigned_anchor(anchors, boxes, imgsize, stride, thres):

    from utils.cython_bbox import bbox_overlaps

    if len(boxes) == 0:
      return [[] for _ in thres]

    height, width = imgsize

    shift_x = np.arange(0, width, stride)
    shift_y = np.arange(0, height, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors        
    A = len(anchors)
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)
    # ---------------------------------------------------------------------

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= 0) &
        (all_anchors[:, 1] >= 0) &
        (all_anchors[:, 2] < width) &  # width
        (all_anchors[:, 3] < height)    # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :].copy()

    if len(boxes) == 0:
      ipdb.set_trace()

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)            
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(boxes, dtype=np.float))           

    argmax_overlaps = overlaps.argmax(axis=1)               # gt index

    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]   # for anchors
    
    gt_argmax_overlaps = overlaps.argmax(axis=0)            # anchor index
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]    # for boxes
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    gt_anchors = anchors[gt_argmax_overlaps, :]

    return [ np.vstack( (anchors[max_overlaps > thr, :], gt_anchors) ) for thr in thres ]
    

def get_assigned_anchor_index(anchors, boxes, imgsize, stride):

    from utils.cython_bbox import bbox_overlaps

    if len(boxes) == 0:
      return []

    height, width = imgsize

    shift_x = np.arange(0, width, stride)
    shift_y = np.arange(0, height, stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors        
    A = len(anchors)
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)
    # ---------------------------------------------------------------------

    # # only keep anchors inside the image
    # inds_inside = np.where(
    #     (all_anchors[:, 0] >= 0) &
    #     (all_anchors[:, 1] >= 0) &
    #     (all_anchors[:, 2] < width) &  # width
    #     (all_anchors[:, 3] < height)    # height
    # )[0]

    # keep only inside anchors
    anchors = all_anchors.copy()

    if len(boxes) == 0:
      ipdb.set_trace()

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)            
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(boxes, dtype=np.float))           

    argmax_overlaps = overlaps.argmax(axis=1)               # gt index

    max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]   # for anchors
    
    gt_argmax_overlaps = overlaps.argmax(axis=0)            # anchor index
    
    return gt_argmax_overlaps % A


def gen_anchors(roidb, num_anchors, valid_cls):
                 
    max_size = cfg.TRAIN.MAX_SIZE
    target_size = cfg.TRAIN.SCALES[0]
    im_size_min, im_size_max = (375, 1242)
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    # anchors = anchors * im_scale

    boxes = []        
    for rr in roidb:
        for cls, box in zip(rr['gt_classes'], rr['boxes']):            
            if cls in valid_cls:                
                box = box * im_scale
                boxes.append(box)                

    boxes = np.vstack( boxes )        
    boxes_wh = np.log( boxes[:,2:] - boxes[:, :2] )

    from sklearn.cluster import KMeans        

    km = KMeans(n_clusters=num_anchors)
    km.fit(boxes_wh)

    # # Show statistics
    # boxes_wh_k = [boxes_wh[km.labels_==l, :] for l in range(num_anchors)]
    # stds = [np.mean((ctr - wh)**2, axis=0) for ctr, wh in zip(km.cluster_centers_, boxes_wh_k)]
    # nSamples = [len(wh) for wh in boxes_wh_k]            

    # Construct anchors ([w_center, h_center] -> [x1 y1 x2 y2])
    wh_centers = np.vstack( (np.exp(km.cluster_centers_)) )

    area = wh_centers[:,0] * wh_centers[:,1]
    idx = area.argsort()
    wh_centers = wh_centers[idx, :]
    anchors = np.hstack( (-1 * wh_centers/2., wh_centers/2.))

    return anchors

if __name__ == '__main__':

    cfg_from_file( os.path.join(cfg.ROOT_DIR, 'experiments', 'cfgs', 'faster_rcnn_end2end_kitti_ZF.yml') )
    # cfg_from_file('../../experiments/cfgs/faster_rcnn_end2end_kitti_vgg16.yml')
    # cfg_from_file('../../experiments/cfgs/faster_rcnn_end2end_kitti_alexnet.yml')

    # from datasets.kitti_raw import kitti
    # imdb = kitti('train', '2012')
    imdb = kitti('trainval', '2012')

    # Apply data augmentation
    imdb.append_flipped_images()        
    # imdb.append_crop_resize_images()    
    # imdb.append_photometric_transformed_images()        

    roidb = imdb.roidb

    plt.ion()
        
    num_anchors = 20

    # anchors_person = gen_anchors(imdb.roidb, 10, [1])
    # anchors_cyclist = gen_anchors(imdb.roidb, 10, [2])
    # anchors_car = gen_anchors(imdb.roidb, 60, [3])
    # anchors = np.vstack( (anchors_person, anchors_cyclist, anchors_car) )    
    
    anchors = gen_anchors(imdb.roidb, num_anchors, [1, 2, 3])
    
    


          
    from rpn.generate_anchors import generate_anchors
    # anchor_scales = np.exp( np.linspace( np.log(2), np.log(11), 3 ) )    
    # anchor_ratios = np.exp( np.linspace( np.log(0.3), np.log(2), 3) )
    anchor_scales = (2, 4, 8, 16, 32)
    anchor_ratios = (0.5, 1, 2.0)
    anchors_ = generate_anchors(scales=np.array(anchor_scales), ratios=np.array(anchor_ratios))
    
    # Draw anchors
    fig = plt.figure(1, figsize=(15,10))
    axes = [ fig.add_subplot(2,1,ii+1) for ii in range(2) ]
    
    clrs = sns.color_palette("Set2", 100)
    axes[0].set_xlim(-200, 200)
    axes[0].set_ylim(-200, 200)
    axes[1].set_xlim(-200, 200)
    axes[1].set_ylim(-200, 200)  
    
    for aa, clr in zip(anchors, clrs):
        axes[0].add_patch( plt.Rectangle( (aa[0], aa[1]), aa[2]-aa[0], aa[3]-aa[1], fill=False, edgecolor=clr, linewidth=3.5) )
        axes[0].axis('equal')
        # plt.pause(0.1)

    for aa, clr in zip(anchors_, clrs):
        axes[1].add_patch( plt.Rectangle( (aa[0], aa[1]), aa[2]-aa[0], aa[3]-aa[1], fill=False, edgecolor=clr, linewidth=3.5) )
        axes[1].axis('equal')
        # plt.pause(0.1)

    plt.pause(1)
    

    np.set_printoptions(precision=2)    

    print anchors

    ipdb.set_trace()

    
    import numpy.random as npr
    import cv2

    img = cv2.imread( imdb.image_path_at(0) )    
    anchor_hist = []
    for rr in imdb.roidb:
      index = get_assigned_anchor_index(anchors, rr['boxes'], img.shape[0:2], 16)
      anchor_hist.extend(index)

    num_assigned_anchors = np.histogram( anchor_hist, 79 )

    
    plt.figure(11)
    plt.hist(anchor_hist, len(anchors))
    plt.pause(1)

    ipdb.set_trace()


    # Draw image-bbox
    idx = npr.choice( imdb.num_images, 30 )

    fig = plt.figure(2, figsize=(30,15))
    axes = [ fig.add_subplot(3,1,ii+1) for ii in range(3) ]
    
    from pprint import pprint
    from transform.image_transform import _flip, _crop_resize, _gamma_correction

    for ii in idx:
        axes[0].cla()
        axes[1].cla()
        axes[2].cla()

        rr = imdb.roidb[ii]

        im = cv2.imread( imdb.image_path_at(ii) )
        im = im[:,:,(2,1,0)]
        
        if rr['flipped']:
            img = _flip(im)
        else:
            img = im.copy()

        if rr['gamma']:
            img = _gamma_correction(img)

        if rr['crop'] is not None:
            img = _crop_resize(img, rr['crop'])

        axes[0].imshow(im)      # original, show assigned anchors with overlap thr = 0.5
        axes[1].imshow(img)     # transformed, show assigned anchors with overlap thr = 0.7
        axes[2].imshow(img)     # transformed, show assigned anchors_ with overlap thr = 0.7

        pprint( rr )
        
        # Draw assigned anchors
        assigned_anchors_1, assigned_anchors_2 = get_assigned_anchor(anchors, rr['boxes'], img.shape[0:2], 16, [0.5, 0.7])
        
        for aa1, aa2 in zip(assigned_anchors_1, assigned_anchors_2):
            # axes[0].add_patch( plt.Rectangle( (aa1[0], aa1[1]), aa1[2]-aa1[0], aa1[3]-aa1[1], fill=False, edgecolor='red',linewidth=2.5) )
            axes[1].add_patch( plt.Rectangle( (aa2[0], aa2[1]), aa2[2]-aa2[0], aa2[3]-aa2[1], fill=False, edgecolor='red',linewidth=2.5) )

        assigned_anchors_2_ = get_assigned_anchor(anchors_, rr['boxes'], img.shape[0:2], 16, [0.7])        

        for aa2 in assigned_anchors_2_[0]:
            axes[2].add_patch( plt.Rectangle( (aa2[0], aa2[1]), aa2[2]-aa2[0], aa2[3]-aa2[1], fill=False, edgecolor='red',linewidth=2.5) )

        # GT
        for bb, cls in zip(rr['boxes'], rr['gt_classes']):
            clr = clrs[cls]
            axes[0].add_patch( plt.Rectangle( (bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, edgecolor=clr,linewidth=2.5) )
            axes[1].add_patch( plt.Rectangle( (bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, edgecolor=clr,linewidth=2.5) )
            axes[2].add_patch( plt.Rectangle( (bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, edgecolor=clr,linewidth=2.5) )

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

        # plt.pause(1)
        plt.savefig('test.jpg')
        
        ipdb.set_trace()

    from IPython import embed; embed()

