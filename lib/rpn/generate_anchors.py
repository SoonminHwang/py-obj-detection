# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def kitti_kmeans_anchors_2x():
    # imdb
    # 'hRng' : [20, np.inf], # Min. 20 x 50 or 25 x 40
    # 'occLevel' : [0, 1, 2],       # 0: fully visible, 1: partly occ, 2: largely occ, 3: unknown
    # 'truncRng' : [0, 0.5]
    
    # # of anchors: 70, scale2x (755x2500)
    anchors = np.array([[  -8.15,  -29.64,    8.15,   29.64],
                       [  -9.92,  -40.11,    9.92,   40.11],
                       [ -12.49,  -31.93,   12.49,   31.93],
                       [ -13.9 ,  -41.98,   13.9 ,   41.98],
                       [ -27.15,  -26.49,   27.15,   26.49],
                       [ -20.32,  -38.53,   20.32,   38.53],
                       [ -16.33,  -48.47,   16.33,   48.47],
                       [ -32.78,  -26.32,   32.78,   26.32],
                       [ -34.69,  -30.79,   34.69,   30.79],
                       [ -41.93,  -26.65,   41.93,   26.65],
                       [ -20.02,  -56.61,   20.02,   56.61],
                       [ -37.38,  -36.64,   37.38,   36.64],
                       [ -52.37,  -26.64,   52.37,   26.64],
                       [ -43.85,  -33.21,   43.85,   33.21],
                       [ -30.62,  -49.52,   30.62,   49.52],
                       [ -23.64,  -68.57,   23.64,   68.57],
                       [ -55.04,  -32.36,   55.04,   32.36],
                       [ -43.8 ,  -40.68,   43.8 ,   40.68],
                       [ -66.16,  -27.09,   66.16,   27.09],
                       [ -52.25,  -39.97,   52.25,   39.97],
                       [ -30.26,  -71.18,   30.26,   71.18],
                       [ -84.36,  -28.73,   84.36,   28.73],
                       [ -50.74,  -48.15,   50.74,   48.15],
                       [ -74.64,  -32.86,   74.64,   32.86],
                       [ -65.67,  -38.54,   65.67,   38.54],
                       [ -29.86,  -88.88,   29.86,   88.88],
                       [ -43.67,  -60.88,   43.67,   60.88],
                       [ -59.69,  -46.12,   59.69,   46.12],
                       [ -88.98,  -35.97,   88.98,   35.97],
                       [ -58.05,  -55.24,   58.05,   55.24],
                       [ -37.9 ,  -90.43,   37.9 ,   90.43],
                       [ -82.17,  -43.13,   82.17,   43.13],
                       [-105.88,  -34.03,  105.88,   34.03],
                       [ -71.18,  -51.39,   71.18,   51.39],
                       [ -69.07,  -63.13,   69.07,   63.13],
                       [-105.57,  -41.79,  105.57,   41.79],
                       [ -60.47,  -75.79,   60.47,   75.79],
                       [ -42.06, -116.46,   42.06,  116.46],
                       [ -82.96,  -59.34,   82.96,   59.34],
                       [ -98.91,  -49.86,   98.91,   49.86],
                       [-131.02,  -43.05,  131.02,   43.05],
                       [ -54.94, -109.33,   54.94,  109.33],
                       [-122.48,  -51.4 ,  122.48,   51.4 ],
                       [ -87.74,  -72.54,   87.74,   72.54],
                       [-102.33,  -63.57,  102.33,   63.57],
                       [ -90.14,  -90.32,   90.14,   90.32],
                       [ -53.54, -152.09,   53.54,  152.09],
                       [-131.56,  -62.5 ,  131.56,   62.5 ],
                       [-156.7 ,  -56.49,  156.7 ,   56.49],
                       [-114.8 ,  -79.97,  114.8 ,   79.97],
                       [ -68.32, -143.39,   68.32,  143.39],
                       [-156.3 ,  -73.9 ,  156.3 ,   73.9 ],
                       [-118.1 , -107.32,  118.1 ,  107.32],
                       [-140.09,  -93.25,  140.09,   93.25],
                       [ -73.64, -179.06,   73.64,  179.06],
                       [-194.04,  -69.6 ,  194.04,   69.6 ],
                       [ -92.52, -183.91,   92.52,  183.91],
                       [-186.74,  -92.41,  186.74,   92.41],
                       [-163.21, -113.73,  163.21,  113.73],
                       [-237.71,  -84.82,  237.71,   84.82],
                       [-118.05, -214.63,  118.05,  214.63],
                       [-213.08, -120.22,  213.08,  120.22],
                       [-180.52, -146.07,  180.52,  146.07],
                       [-274.61, -102.48,  274.61,  102.48],
                       [-250.69, -151.21,  250.69,  151.21],
                       [-332.59, -126.28,  332.59,  126.28],
                       [-214.03, -206.52,  214.03,  206.52],
                       [-298.27, -184.74,  298.27,  184.74],
                       [-396.23, -182.24,  396.23,  182.24],
                       [-370.13, -252.58,  370.13,  252.58]])
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
