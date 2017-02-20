#!/usr/bin python

### VOC-stype dataset converter
# Extract jpeg images from .seq file
# Write a text file including file names for Caltech pedestrian datasets

import os
import glob
import cv2 as cv
import re
import argparse

def parse_argument():
	"""
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Dataset-stype converter for Caltech')
    parser.add_argument('--srcDir', dest='srcDir',
                        help='Source video directory including .seq files',
                        default='data/caltech/videos/', type=str)
    parser.add_argument('--imgDir', dest='imgDir',
                        help='Destination of images',
                        default='data/caltech/images/', type=str)
    parser.add_argument('--txtDir', dest='txtDir',
                        help='Text files that specifies train/test sets',
                        default='data/caltech/imageSets/', type=str)
	
	parser.add_argument('--type', dest='type',
                        help='train or test',
                        type=str)

	parser.add_argument('--skip', dest='skip',
                        help='# of skipped frames',
                        default=30, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def save_img(dstDir, vidName, i, frame):
	exps = re.findall('*set(\d+)*V(\d+)*', vidName)
	s = exps.group(1)
	v = exps.group(2)

	imgNm = os.path.join('set{:02d}'.format(s), 'V{:03d}'.format(v), 'I{:06d}'.format(i))	
    cv.imwrite(os.path.join(dstDir, imgNm + '.png'), frame)

    return imgNm

if __name__ == '__main__':

	args = parse_args()

	textFile = os.path.join(args.txtDir, '{s}{:02d}.txt'.format(args.type, args.skip))
	f = fopen(textFile, 'w')
	for set_name in sorted(glob.glob(args.srcDir + 'set*')):
    	for vid_name in sorted(glob.glob('{}/*.seq'.format(set_name))):
    		print('Extract video file (.seq): {s}'.format(vid_name))
	        cap = cv.VideoCapture(vid_name)
	        i = 0
	        while True:
	            ret, frame = cap.read()
	            if not ret: break
	            imgNm = save_img(args.imgDir, vid_name, i, frame)
	            f.write(imgNm)
	            i += 1
	        
	f.close()