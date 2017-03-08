#!/home/rcvlab/anaconda2/bin/python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys

from make_models import write_solver

import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)    
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb_train', dest='imdb_train_name',
                        help='dataset to train on',
                        default='kitti_2012_train', type=str)
    parser.add_argument('--imdb_val', dest='imdb_val_name',
                        help='dataset to validation on',
                        default='kitti_2012_val', type=str)
    parser.add_argument('--log_dir', dest='log_dir',
                        help='a path to save snapshots',
                        default='', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names, isTrain):
    def get_roidb(imdb_name, isTrain):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb, isTrain)
        return roidb

    roidbs = [get_roidb(s, isTrain) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':        
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb_train, roidb_train = combined_roidb(args.imdb_train_name, True)
    imdb_val, roidb_val = combined_roidb(args.imdb_val_name, False)

    print '[Train] {:d} roidb entries'.format(len(roidb_train))
    print '[Val] {:d} roidb entries'.format(len(roidb_val))

    # output_dir = get_output_dir(imdb)
    output_dir = os.path.join( args.log_dir, 'snapshots' )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)        
    print 'Output will be saved to `{:s}`'.format(output_dir)

    snapshot_prefix = os.path.join(output_dir, cfg.SOLVER.SNAPSHOT_PREFIX)
    net = os.path.join(args.log_dir, 'models', 'trainval.prototxt')

    solver_file = write_solver(args.log_dir, base_lr=cfg.SOLVER.BASE_LR, lr_policy=cfg.SOLVER.LR_POLICY, 
        gamma=cfg.SOLVER.GAMMA, stepsize=cfg.SOLVER.STEPSIZE, display=cfg.SOLVER.DISPLAY, 
        average_loss=cfg.SOLVER.AVERAGE_LOSS, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY, 
        snapshot=cfg.SOLVER.SNAPSHOT, iter_size=cfg.SOLVER.ITER_SIZE, 
        test_iter=cfg.SOLVER.TEST_ITER, test_interval=cfg.SOLVER.TEST_INTERVAL,
        test_compute_loss=cfg.SOLVER.TEST_COMPUTE_LOSS, test_initialization=cfg.SOLVER.TEST_INITIALIZATION, 
        max_iter=cfg.SOLVER.MAX_ITER,
        snapshot_prefix=snapshot_prefix, net=net)

    train_net(solver_file, roidb_train, roidb_val, imdb_train, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=cfg.SOLVER.MAX_ITER)
