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
from easydict import EasyDict as edict
from utils.cython_bbox import bbox_overlaps


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
    * top[0] - list of bbox
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
            param_str : "{'iou_thres': [0.7, 0.5, 0.5]}"
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

    boxes = proposal[:, 1:5] / im_info[0, 2]

    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_info[0, :2])


    results = np.zeros((0, 6), dtype=np.float32)

    for cls_ind in xrange(cfg.NET.NUM_CLASSES-1):
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


class EvalLayer(caffe.Layer):
    """
    * Accumulate bbox lists for N (=cfg.SOLVER.TEST_ITER) images
    * Then, calculates mean average precision

    * bottom[0] - marked up bbox list (prediction)
        [ num_boxes_after_nms x 6 (cls_ind, xl, yt, xr, yb, class) ]
    * bottom[1] - gt_boxes
        [ num_boxes x 5 (xl, yt, xr, yb, cls_ind) ]
    * top[0] - mAP
        [ 1 x 3 (car, pedestrian, cyclist) x 3 (easy, moderate, hard) ]

    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'mAP'
        top: 'mAP_Car'
        top: 'mAP_Ped'
        top: 'mAP_Cyc'        
        bottom: 'bbox_list'
        bottom: 'gt_boxes'
        python_param {
            module: 'eval.detectnet'
            layer: 'EvalLayer'
            # parameters (default)
            #   - DIFFICULTY = {'easy': 0, 'moderate': 1, 'hard': 2}
            #   - MIN_HEIGHT = (40, 25, 25)
            #   - MAX_OCCLUSION = (0, 1, 2)
            #   - MAX_TRANCATION = (0.15, 0.3, 0.5)
            #   - CLASSES = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
            #   - NEIGHBOR_CLASSES = {'Car': ['Van'], 'Pedestrian': ['Person_sitting'], 'Cyclist': []]}
            #   - MIN_OVERLAP = (0.7, 0.5, 0.5)
            #   - N_SAMPLE_PTS = 41
            # Ex) param_str : "{MIN_OVERLAP: (0.7, 0.5, 0.5)}
            param_str : ""
        }
        include: { phase: TEST }
    }
    """

    def update_eval_params(self, param, new):
        for key, val in new.items():
            if key in param:
                param[key] = val

        return param

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)            

        p = edict()
        p.DIFFICULTY = {'easy': 0, 'moderate': 1, 'hard': 2}
        p.MIN_HEIGHT = (40, 25, 25)           # minimum height for evaluated groundtruth/detections
        p.MAX_OCCLUSION = (0, 1, 2)           # maximum occlusion level of the groundtruth used for evaluation
        p.MAX_TRUNCATION = (0.15, 0.3, 0.5)   # maximum truncation level of the groundtruth used for evaluation
        
        p.CLASSES = {'Car': 3, 'Pedestrian': 1, 'Cyclist': 2} 
        p.NEIGHBOR_CLASSES = {'Car': ['Van'], 'Pedestrian': ['Person_sitting'], 'Cyclist': []}
        p.MIN_OVERLAP = (0.5, 0.5, 0.7)       # the minimum overlap required for evaluation
        p.N_SAMPLE_PTS = 41   # number of recall steps that should be evaluated (discretized)

        self.p = self.update_eval_params(p, layer_params)
        
        self._num_cur_inputs = 0
        self.pred = [ np.zeros( (0, 10), np.float32 ) for _ in xrange(len(p.CLASSES)) ]
        self.gt = [ np.zeros( (0, 9), np.float32 ) for _ in xrange(len(p.CLASSES)) ]


    def reshape(self, bottom, top):
        # top_shape = len(self.p.CLASSES), len(self.p.DIFFICULTY)
        # top[0].reshape(*top_shape)
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        

    def forward(self, bottom, top):          
        # gts: [ curIdx, bTP, x1, y1, x2, y2, cls_ind, occ, trunc ]
        # dts: [ curIdx, bTP, cls_ind, x1, y1, x2, y2, score ]

        gts, dts = match(self, self.p, bottom[1].data.copy(), bottom[0].data.copy(), bottom[3].data.copy())
        self.pred = [ np.vstack((p_old, p_new)) for p_old, p_new in zip(self.pred, dts) ]
        self.gt = [ np.vstack((g_old, g_new)) for g_old, g_new in zip(self.gt, gts) ]

        # ## DEBUG
        # img = bottom[2].data[0].copy()
        # img = img.transpose((1,2,0))            
        # img += cfg.PIXEL_MEANS    
        # # img = img[:,:,(2,1,0)]
        # img = img.astype(np.uint8)

        # im_info = bottom[3].data.copy()

        # import cv2
        # im_scale_inv = 1.0 / im_info[0,2]        
        # img = cv2.resize(img, None, None, fx=im_scale_inv, fy=im_scale_inv,
        #             interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite('tmp/image_%03d.jpg' % self._num_cur_inputs, img)

        self._num_cur_inputs = self._num_cur_inputs + 1

        if self._num_cur_inputs == cfg.SOLVER.TEST_ITER:
        # if True:

            # np.save('tmp/gt_%03d.npy' % self._num_cur_inputs, self.gt)
            # np.save('tmp/pred_%03d.npy' % self._num_cur_inputs, self.pred)

            # Do evaluation
            if len(bottom) > 2:
                mAP = eval(self, self.gt, self.pred, bottom[2].data.copy(), bottom[3].data.copy())
            else:
                mAP = eval(self, self.gt, self.pred)

            try:                                
                print 'mAP_Ped = %.4f' % mAP[0]
                print 'mAP_Cyc = %.4f' % mAP[1]
                print 'mAP_Car = %.4f' % mAP[2]

                top[0].data[...] = mAP[0] * cfg.SOLVER.TEST_ITER
                top[1].data[...] = mAP[1] * cfg.SOLVER.TEST_ITER
                top[2].data[...] = mAP[2] * cfg.SOLVER.TEST_ITER
            except:
                import ipdb     
                ipdb.set_trace()
    
            self.pred = [ np.zeros( (0, 10), np.float32 ) for _ in xrange(len(self.p.CLASSES)) ]
            self.gt = [ np.zeros( (0, 9), np.float32 ) for _ in xrange(len(self.p.CLASSES)) ]            

            self._num_cur_inputs = 0
        else:        
            top[0].data[...] = 0
            top[1].data[...] = 0
            top[2].data[...] = 0
            

    def backward(self, top, propagate_down, bottom):
        pass



def match(self, params, gts, dts, im_info):
    # gts: [ curIdx, bTP, x1, y1, x2, y2, cls_ind, occ, trunc ]
    # dts: [ curIdx, bTP, cls_ind, x1, y1, x2, y2, score, occ, trunc ]

    gts[:, 0:4] = gts[:, 0:4] / im_info[0,2]
    
    gt_boxes = np.insert(gts, 0, 0, axis=1)
    dt_boxes = np.insert(dts, 0, 0, axis=1)

    gt_boxes = np.insert(gt_boxes, 0, self._num_cur_inputs, axis=1)
    dt_boxes = np.insert(dt_boxes, 0, self._num_cur_inputs, axis=1)

    dt_boxes = np.hstack( (dt_boxes, np.zeros( (len(dt_boxes), 2) ) ) )

    gt_matched = [ [] for _ in xrange( len(params.CLASSES) ) ]
    dt_matched = [ [] for _ in xrange( len(params.CLASSES) ) ]

    for cls, cls_ind in params.CLASSES.items():

        # 0. Filtering (just small detection)        
        gt_valid = gt_boxes[:,6] == cls_ind
        dt_valid = dt_boxes[:,2] == cls_ind

        gt_cls = gt_boxes[gt_valid, : ]
        dt_cls = dt_boxes[dt_valid, : ]

        # 1. Check TP, FP, FN
        # iou overlaps        
        overlaps = np.zeros( (len(dt_cls), len(gt_cls)), dtype=np.float32 )
        for ii, g in enumerate(gt_cls[:,2:6]):
            for jj, d in enumerate(dt_cls[:,3:7]):                
                overlaps[jj, ii] = iou(d, g)

        gt_tp_flag = np.zeros((len(gt_cls)), np.float32)
        dt_tp_flag = np.zeros((len(dt_cls)), np.float32)

        if len(gt_cls) != 0 and len(dt_cls) != 0:            
            dt_argmax_overlaps = overlaps.argmax(axis=1)
            dt_max_overlaps = overlaps[np.arange(len(overlaps)), dt_argmax_overlaps]
                    
            dt_tp_flag[ dt_max_overlaps > params.MIN_OVERLAP[cls_ind-1] ] = 1
            dt_cls[:, 1] = dt_tp_flag
            dt_cls[:, 8:] = gt_cls[ dt_argmax_overlaps, 8: ]
            

            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]            
            gt_tp_flag[ gt_max_overlaps > params.MIN_OVERLAP[cls_ind-1] ] = 1
            gt_cls[:, 1] = gt_tp_flag
            

        gt_matched[cls_ind-1] = gt_cls
        dt_matched[cls_ind-1] = dt_cls
            
    return gt_matched, dt_matched


def iou(det, rhs):
    x_overlap = max(0, min(det[2], rhs[2]) - max(det[0], rhs[0]))
    y_overlap = max(0, min(det[3], rhs[3]) - max(det[1], rhs[1]))
    overlap_area = x_overlap * y_overlap
    if overlap_area == 0:
        return 0
    det_area = (det[2]-det[0])*(det[3]-det[1])
    rhs_area = (rhs[2]-rhs[0])*(rhs[3]-rhs[1])
    unionarea = det_area + rhs_area - overlap_area
    return overlap_area/unionarea

def eval(self, gts, dts, image=None, im_info=None):
    """
        gts: N x [ curIdx, bTP, x1, y1, x2, y2, cls_ind ]
        dts: M x [ curIdx, bTP, cls_ind, x1, y1, x2, y2, score ]        
    """

    # TODO(smh): Add trunc/occ values for each gt_boxes 
    # TODO(smh): Then, evaluate with each difficulty conditions
    # TODO(smh): For more accurate evaluation, don't care objects must be considered
    # TODO(smh): Add Easy/Hard
            
    ## DEBUG
    # if image is not None:
    #     import matplotlib.pyplot as plt
    #     img = image[0].copy()
    #     img = img.transpose((1,2,0))            
    #     img += cfg.PIXEL_MEANS    
    #     img = img[:,:,(2,1,0)]
    #     img = img.astype(np.uint8)

    #     import cv2
    #     im_scale_inv = 1.0 / im_info[0,2]        
    #     img = cv2.resize(img, None, None, fx=im_scale_inv, fy=im_scale_inv,
    #                 interpolation=cv2.INTER_LINEAR)

    #     import matplotlib.pyplot as plt
    #     plt.figure(2)
    #     plt.clf()
    #     plt.ion()
    #     plt.imshow( img )
    #     axe = plt.gca()

    #     for gt in gts:
    #         for g in gt[ gt[:,0] == self._num_cur_inputs-1, :]:
    #             axe.add_patch( plt.Rectangle((g[2], g[3]), g[4]-g[2], g[5]-g[3], fill=False, edgecolor='r', linewidth=2) )
        
    #     import seaborn as sns
    #     clrs = sns.color_palette("Set2", 5)
    #     for dt in dts:
    #         for d in dt[ dt[:,0] == self._num_cur_inputs-1, :]:
    #             clr = clrs[ int(d[2]) ]
                
    #             if d[-3] > 0.5:
    #                 axe.add_patch( plt.Rectangle((d[3], d[4]), d[5]-d[3], d[6]-d[4], fill=False, edgecolor=clr, linewidth=1.5) )
    #                 axe.text(d[3], d[4]-2, '{:.3f}'.format(d[-3]), 
    #                     bbox=dict(facecolor=clr, alpha=0.5), fontsize=8, color='white')

    #     plt.axis('off')
    #     plt.savefig('EvalLayer_input.jpg', dpi=200)

    mAP = []
    
    for gt_cls, dt_cls in zip(gts, dts):
        # sorting by scores
        order = np.argsort(dt_cls[:,-3])
        dt_cls = dt_cls[order[::-1], : ]
        
        h_gt = gt_cls[:,5] - gt_cls[:,3] + 1
        h_dt = dt_cls[:,6] - dt_cls[:,4] + 1

        dt_occ = dt_cls[:,-2]
        dt_trunc = dt_cls[:,-1]

        P = len(gt_cls)
        D = len(dt_cls)

        if P == 0:
            print( 'No GT instance. Increase TEST_ITER value.')
            import ipdb
            ipdb.set_trace()

        if D == 0:
            mAP.append(0.0)
            continue
        
        # 2. Evaluation for each difficulties
        # for diff_name, diff_ind in params.DIFFICULTY.items():
        d = 1    # Moderate
                
        cond = ( h_dt >= self.p.MIN_HEIGHT[d] ) & ( dt_occ <= self.p.MAX_OCCLUSION[d] ) & ( dt_trunc <= self.p.MAX_TRUNCATION[d] )
        dt_cls_diff = dt_cls[ cond, : ]

        # dt_cls_diff = dt_cls[ h_dt >= params.MIN_HEIGHT[diff_ind], : ]
        dt_tp_scores = dt_cls_diff[ dt_cls_diff[:,1] == 1, -3 ]
        scores = dt_cls_diff[:,-3]

        # compute thresholds for each recall        
        cur_recall = 0.0
        thresholds = np.zeros( (self.p.N_SAMPLE_PTS - 1.0), dtype=np.float32 )
        tt = 0

        for ii, score in enumerate(dt_tp_scores):
            l_recall = (ii+1) / float(P)
            r_recall = (ii+2) / float(P) if ii < len(dt_tp_scores)-1 else l_recall

            if (r_recall - cur_recall) < (cur_recall - l_recall) and ii < len(dt_tp_scores)-1:
                continue

            # left recall is the best approximation, so use this and goto next recall step for approximation
            recall = l_recall

            thresholds[tt] = score
            tt += 1
            cur_recall += 1.0 / (self.p.N_SAMPLE_PTS - 1.0)

        # compute precision/recall for each threshold
        prec = np.zeros( len(thresholds), np.float32 )
        for ii, thr in enumerate(thresholds):
            dt_thres = dt_cls_diff[ scores >= thr, :]
            prec[ii] = np.sum( dt_thres[:,1] ) / len( dt_thres )
                    
        mAP.append( np.mean(prec) * 100 )

    return mAP


# def iou(det, rhs):
#     x_overlap = max(0, min(det[2], rhs[2]) - max(det[0], rhs[0]))
#     y_overlap = max(0, min(det[3], rhs[3]) - max(det[1], rhs[1]))
#     overlap_area = x_overlap * y_overlap
#     if overlap_area == 0:
#         return 0
#     det_area = (det[2]-det[0])*(det[3]-det[1])
#     rhs_area = (rhs[2]-rhs[0])*(rhs[3]-rhs[1])
#     unionarea = det_area + rhs_area - overlap_area
#     return overlap_area/unionarea


# def divide_zero_is_zero(a, b):
#     return float(a)/float(b) if b != 0 else 0


# def score_det(gt_bbox_list, det_bbox_list):
#     threshold = 0.7
#     matched_bbox = np.zeros([gt_bbox_list.shape[0], MAX_BOXES, 5])

#     for k in range(gt_bbox_list.shape[0]):

#         # Remove  zeros from detected bboxes
#         cur_det_bbox = det_bbox_list[k, :, 0:4]
#         cur_det_bbox = np.asarray(filter(lambda a: a.tolist() != [0, 0, 0, 0], cur_det_bbox))

#         # Remove  zeros from label bboxes
#         cur_gt_bbox = gt_bbox_list[k, :, 0:4]
#         cur_gt_bbox = np.asarray(filter(lambda a: a.tolist() != [0, 0, 0, 0], cur_gt_bbox))

#         gt_matched = np.zeros([cur_gt_bbox.shape[0]])
#         det_matched = np.zeros([cur_det_bbox.shape[0]])

#         for i in range(cur_gt_bbox.shape[0]):
#             for j in range(cur_det_bbox.shape[0]):
#                 if (iou(cur_det_bbox[j], cur_gt_bbox[i]) >= threshold) and (det_matched[j] == 0):
#                     gt_matched[i] = 1
#                     det_matched[j] = 1
#                     break

#         tp = np.asarray([np.append(j, 1) for j in cur_det_bbox[np.where(det_matched == 1)]])
#         fp = np.asarray([np.append(j, 2) for j in cur_det_bbox[np.where(det_matched == 0)]])
#         tn = np.asarray([np.append(j, 3) for j in cur_gt_bbox[np.where(gt_matched == 0)]])

#         temp = np.append(tp, fp)
#         temp = np.append(temp, tn)
#         temp = temp.reshape([temp.size/5, 5])
#         matched_bbox[k, 0:temp.shape[0], :] = temp

#     return matched_bbox


# def calcmAP(scored_detections, self):
#     self.true_positives = np.where(scored_detections[:, :, 4] == 1)[0].size
#     self.false_positives = np.where(scored_detections[:, :, 4] == 2)[0].size
#     self.true_negatives = np.where(scored_detections[:, :, 4] == 3)[0].size
#     self.precision = divide_zero_is_zero(self.true_positives, self.true_positives+self.false_positives)*100.00
#     self.recall = divide_zero_is_zero(self.true_positives, self.true_positives+self.true_negatives)*100.00
#     self.avp = self.precision * self.recall / 100.0


if __name__ == '__main__':

    import caffe
    net = caffe.Net('eval/layerTest.prototxt', caffe.TEST)
    net.forward()