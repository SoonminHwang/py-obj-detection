import glob
import re
import os

models = glob.glob('snapshots/zf_iter_*.caffemodel')

for model in models:
	rstPath = model.split('/')[1].split('.')[0]
	if os.path.exists('results/' + rstPath):
		continue

	cmd = '../../../../tools/test_net.py --gpu 0 --def "${CUR}/models/test.prototxt" --cfg "${CUR}/faster_rcnn_end2end_kitti_ZF.yml" --imdb kitti_val --net "${CUR}/%s" --output_dir "${CUR}/results/%s"' % (model, rstPath)

	os.system(cmd)


	
