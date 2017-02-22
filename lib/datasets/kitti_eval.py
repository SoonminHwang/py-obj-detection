### Re-implementation of evaluate_object.cpp in devkit
### 07.02.11.
### Written by Soonmin Hwang

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from cycler import cycler

import ipdb

# import matplotlib
# matplotlib.rcParams.update({'font.size': 22})

class EvalKITTI(object):

	DIFFICULTY = {'easy': 0, 'moderate': 1, 'hard': 2}
	MIN_HEIGHT = (40, 25, 25)			# minimum height for evaluated groundtruth/detections
	MAX_OCCLUSION = (0, 1, 2)			# maximum occlusion level of the groundtruth used for evaluation
	MAX_TRUNCATION = (0.15, 0.3, 0.5)	# maximum truncation level of the groundtruth used for evaluation

	CLASSES = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}	
	NEIGHBOR_CLASSES = {'Car': ['Van'], 'Pedestrian': ['Person_sitting'], 'Cyclist': []}
	MIN_OVERLAP = (0.7, 0.5, 0.5)		# the minimum overlap required for evaluation
	N_SAMPLE_PTS = 41	# number of recall steps that should be evaluated (discretized)

	def __init__(self, gtPth, rstPth, plotPth=None, basePth=None):

		if basePth is None:
			basePth = os.path.abspath( os.path.join( os.path.dirname(__file__), '..' ) )
			
		self._gtPth = os.path.join( basePth, gtPth )
		self._rstPth = os.path.join( basePth, rstPth )
		self._plotPth = os.path.join(rstPth, 'plots') if plotPth is None else plotPth

		
	def evaluate_in_layer(self, gts_list, dts_list, ind_to_class):
		gts = []
		dts = []

		for gt, dt in zip(gts_list, dts_list):
			gts.append( self._load_gt_from_bbox(gt, ind_to_class) )
			dts.append( self._load_dt_from_bbox(dt, ind_to_class) )

		# Class-wise evaluation (car, pedestrian, cyclist)		
		stats = {}
		for cls_str, cls_idx in self.CLASSES.items():

			stats[cls_str] = {}
			#fp = open( os.path.join(self._plotPth, cls_str + '_detection.txt'), 'w' )
			
			# Evaluation conditions (easy, moderate, hard)
			for cond_str, cond_idx in self.DIFFICULTY.items():
				# precision = self._eval_class(gts, dts, cls_str, cond_idx, fp)
				precision = self._eval_class(gts, dts, cls_str, cond_idx)
				stats[cls_str][cond_str] = precision

			# Make plots
			plot_file_name = os.path.join(self._plotPth, cls_str + '_detection')
			self._make_plots(cls_str, stats[cls_str], plot_file_name)


	def evaluate(self):
		# Init global params
		gts = []
		dts = []
		
		curPth = os.path.abspath('.')
		#rstPth = self._rstPth.replace('[', '\[').replace(']', '\]').replace('/', '\/')
		os.chdir( self._rstPth )
		# get gt_dir, result_dir, plot_dir
		# result_files = glob.glob( os.path.join(rstPth, '*.txt') )
		result_files = glob.glob( '*.txt' )
		os.chdir( curPth ) 

		# files in result_dir, for each file
		for dt_file in result_files:
			
			# file = os.path.basename(dt_file)
			file = dt_file
			dt_file = os.path.join(self._rstPth, file)
			gt_file = os.path.join(self._gtPth, file)

			# Load GTs			
			gts.append( self._load_groundtruths(gt_file) )
			# Load DTs			
			dts.append( self._load_detections(dt_file) )		

		
		if not os.path.exists(self._plotPth): os.makedirs(self._plotPth)

		
		# Class-wise evaluation (car, pedestrian, cyclist)		
		stats = {}
		for cls_str, cls_idx in self.CLASSES.items():

			stats[cls_str] = {}
			#fp = open( os.path.join(self._plotPth, cls_str + '_detection.txt'), 'w' )
			
			# Evaluation conditions (easy, moderate, hard)
			for cond_str, cond_idx in self.DIFFICULTY.items():
				# precision = self._eval_class(gts, dts, cls_str, cond_idx, fp)
				precision = self._eval_class(gts, dts, cls_str, cond_idx)
				stats[cls_str][cond_str] = precision

			# Make plots
			plot_file_name = os.path.join(self._plotPth, cls_str + '_detection')
			self._make_plots(cls_str, stats[cls_str], plot_file_name)
			

	def _make_plots(self, cls, prec, file_name):

		try:
			# Save plot data to file		
			with open(file_name + '.txt', 'w') as f:
				for ii in range(self.N_SAMPLE_PTS):					
					res = tuple( [float(ii)/(self.N_SAMPLE_PTS-1.0)] + [ prec[key][ii] for key in self.DIFFICULTY.keys() ])
					f.write( '%f %f %f %f\n' % res )

				# Draw plots
				N = len(self.DIFFICULTY.keys())
				# fig = plt.figure(figsize=(15,15))
				fig = plt.figure()
				axes = fig.add_subplot(1, 1, 1)

				# axes.set_color_cycle( sns.color_palette("Set2", 3) )		# Deprecated
				axes.set_prop_cycle( cycler('color', sns.color_palette("Set2", 3) ) )

				print '------------------------------------------------------------------------'
				print 'Class: %s' % cls
				print '------------------------------------------------------------------------'

				recall = [ float(jj)/(self.N_SAMPLE_PTS-1.0) for jj in range(self.N_SAMPLE_PTS) ]
				for ii, key in enumerate(self.DIFFICULTY.keys()):
					lgd_str = '%s, %.2f' % (key, np.mean(prec[key]) * 100)
					print '[%s] %.2f' % (key, np.mean(prec[key]) * 100)
					axes.plot( recall, prec[key], '+-', linewidth=3, label=lgd_str )

				print '------------------------------------------------------------------------\n'

				plt.title('Class: %s' % cls)
				plt.legend(loc='best')
				plt.savefig(file_name+'.png', dpi=200)

		except:
			ipdb.set_trace()
				

	def _eval_class(self, gts, dts, cls, difficulty, fp=None):

		result = {}

		ign_gts = []
		ign_dts = []
		dcs = []

		v = []

		tp, fp, fn = (0, 0, 0)
		n_gt = 0
		for gt, dt in zip(gts, dts):
			# clean data			
			ign_gt, ign_dt, dc = self._clean_data(gt, dt, cls, difficulty)
			
			ign_gts.append(ign_gt)
			ign_dts.append(ign_dt)
			dcs.append(dc)

			# compute statistics
			stat_tmp = self._compute_statistics(cls, gt, dt, ign_gt, ign_dt, dc, False)

			tp += stat_tmp.tp
			fp += stat_tmp.fp
			fn += stat_tmp.fn

			n_gt += np.sum([1 if ig == 0 else 0 for ig in ign_gt])

			# add detection scores over all images
			for _v in stat_tmp.v:
				v.append(_v)

		# Compute thresholds
		thres = self._get_thresholds(v, n_gt)

		# precision = [[] for _ in range(len(thres))]
		pr = [ self.tPrData() for _ in range(len(thres))]
		
		for gt, dt, dc, ign_gt, ign_dt in zip(gts, dts, dcs, ign_gts, ign_dts):
			for t, thr in enumerate(thres):			
				# compute statistics
				stat = self._compute_statistics(cls, gt, dt, ign_gt, ign_dt, dc, True, thr)
								
				pr[t].tp += stat.tp
				pr[t].fp += stat.fp
				pr[t].fn += stat.fn

		# Compute precision, recall
		precision = [ 0.0 for _ in range(self.N_SAMPLE_PTS) ]
		# recall = [ 0.0 for _ in range(N_SAMPLE_PTS) ]
		recall = []
		
		for t, thr in enumerate(thres):
			r = pr[t].tp / float(pr[t].tp + pr[t].fn)
			recall.append(r)
			precision[t] = pr[t].tp / float(pr[t].tp + pr[t].fp)

		# # Save statistics
		# for p in precision:	fp.write( '%f' % p )
		# fp.write('\n')

		return precision


	def _boxoverlap(self, a, b, criterion=-1):
		o = -1

		x1, y1, x2, y2 = ( max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]) )
		w, h = (x2-x1, y2-y1)

		if w <= 0 or h <= 0: return 0

		inter = w * h
		a_area = (a[2]-a[0]) * (a[3]-a[1])
		b_area = (b[2]-b[0]) * (b[3]-b[1])

		if criterion == -1:		# union
			o = inter / (a_area + b_area - inter)
		elif criterion == 0:	# bbox_a
			o = inter / a_area
		elif criterion == 1:	# bbox_b
			o = inter / b_area

		return o


	def _get_thresholds(self, v, n_gt):

		thres = []

		v.sort(reverse=True)
		n_gt = float(n_gt)

		# check if right-hand-side recall with respect to current recall is close than left-hand-side one
		# in this case, skip the current detection score
		cur_recall = 0.0
		for ii, score in enumerate(v):
			l_recall = (ii+1) / n_gt
			r_recall = (ii+2) / n_gt if ii < len(v)-1 else l_recall

			if (r_recall - cur_recall) < (cur_recall - l_recall) and ii < len(v)-1:
				continue

			# left recall is the best approximation, so use this and goto next recall step for approximation
			recall = l_recall

			# print( 'Thresholds: %f' % score )

			thres.append( score )
			cur_recall += 1.0 / (self.N_SAMPLE_PTS - 1.0)

		return thres
    	

	def _clean_data(self, gt, dt, cls, difficulty):

		ign_gt = []
		ign_dt = []
		dc = []
		
		for g in gt:
			g_height = g.box[3] - g.box[1]

			# Class label
			g_cls = g.type
			if g_cls == cls:
				valid = 1
			elif g_cls in self.NEIGHBOR_CLASSES[cls]:
				# Ignore neighboring classes (person_sitting, van)
				valid = 0
			else:
				valid = -1
			
			# Check difficulty condition
			g_occ = g.occlusion
			g_trunc = g.truncation
			if g_height < self.MIN_HEIGHT[difficulty] or g_occ > self.MAX_OCCLUSION[difficulty] or g_trunc > self.MAX_TRUNCATION[difficulty]:
				ignore = True
			else:
				ignore = False

			# Set ignore flag
			if valid == 1 and ignore == False:				
				ign_gt.append(0)	# evaluated
			elif valid == 0 or (valid == 1 and ignore == True):
				ign_gt.append(1)	# ignored
			else:
				ign_gt.append(-1)	# FN in the evaluation

			# Load 'DontCare' instances
			if g_cls == 'DontCare':
				dc.append(g)
			
		for d in dt:
			d_cls = d.type

			if d_cls == cls:
				ign_dt.append(0)
			else:
				ign_dt.append(-1)

		return ign_gt, ign_dt, dc
		

	def _compute_statistics(self, cls_str, gt, dt, ign_gt, ign_dt, dc, bFp, thr=0.0):
		
		cls = self.CLASSES[cls_str]

		stat = self.tPrData()		
		assigned_det = [False for _ in dt]

		# Detections with a low score are ignored for computing precision
		# ignored_thres = []
		# if bFp:
		ignored_thres = [True if bFp and d.thresh < thr else False for d in dt]

		# Evaluate all GT boxes
		for g, ign_g in zip(gt, ign_gt):

			# this ground truth is not of the current or a neighboring class and therefore ignored
			if ign_g == -1: continue

			##############################################################################
			# find candidates (overlap with ground truth > 0.5) (logical len(det))
			##############################################################################
			valid_det = -np.inf
			max_overlap = 0.0
			det_idx = -1
			assigned_ign = False
			
			for jj, (d, ign_d, assigned, ignored_thr) in enumerate(zip(dt, ign_dt, assigned_det, ignored_thres)):
				# detections not of the current class, already assigned or with a low threshold are ignored
				if ign_d == -1 or assigned == True or ignored_thr == True: continue
				
				overlap = self._boxoverlap(d.box, g.box)						

				if not bFp and overlap > self.MIN_OVERLAP[cls] and d.thresh > valid_det:
					# for computing recall thresholds, the candidate with highest score is considered
					det_idx = jj
					valid_det = d.thresh
				
				elif bFp and overlap > self.MIN_OVERLAP[cls] and ( overlap > max_overlap or assigned_ign ) and ign_d == 0:
					# for computing pr curve values, the candidate with the greatest overlap is considered
      				# if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
					max_overlap = overlap
					det_idx = jj
					valid_det = 1
					assigned_ign = False
				elif bFp and overlap > self.MIN_OVERLAP[cls] and valid_det == -np.inf and ign_d == 1:
					det_idx = jj
					valid_det = 1
					assigned_ign = True

			##############################################################################
			# compute TP, FP and FN
			##############################################################################
			if valid_det == -np.inf and ign_g == 0:
				# nothing was assigned to this valid ground truth
				stat.fn += 1
			elif valid_det != -np.inf and ( ign_g == 1 or ign_dt[det_idx] == 1 ):
				# only evaluate valid ground truth <=> detection assignments (considering difficulty level)
				assigned_det[det_idx] = True
			elif valid_det != -np.inf:
				# found a valid true positive
				stat.tp += 1
				stat.v.append( dt[det_idx].thresh )
				
				assigned_det[det_idx] = True

		# if FP are requested,
		if bFp:
			for ign_d, ign_thr, assigned in zip(ign_dt, ignored_thres, assigned_det):
				if not (assigned or ign_d == -1 or ign_d == 1 or ign_thr):
					stat.fp += 1

			nstuff = 0
			for c in dc:
				for jj, (d, ign_d, ign_thr, assigned) in enumerate(zip(dt, ign_dt, ignored_thres, assigned_det)):
					if assigned or ign_d == -1 or ign_d == 1 or ign_thr: continue

					overlap = self._boxoverlap(d.box, c.box, 0)
					if overlap > self.MIN_OVERLAP[cls]:
						assigned_det[jj] = True
						nstuff += 1

			stat.fp -= nstuff

		return stat

	def _load_gt_from_bbox(self, gt, ind_to_class):
		groundtruths = []
		for g in gt:
			groundtruths.append( self.tGT(ind_to_class[g[-1]], g[0], g[1], g[2], g[3], 0, 0, 0) )

		return groundtruths


	def _load_dt_from_bbox(self, dt, ind_to_class):
		detections = []
		for d in dt:
			detections.append( self.tDT(ind_to_class[d[0]], d[1], d[2], d[3], d[4], 0, d[5]) )

		return detections


	def _load_groundtruths(self, file_name):
		try:
			with open(file_name, 'r') as f:
				lines = [ line.rstrip('\n').split(' ') for line in f.readlines() ]
				groundtruths = []
				for line in lines:
					cls_str = line[0]
					trunc, occ, alpha = [ float(item) for item in line[1:4] ]
					x1, y1, x2, y2 = [ float(item) for item in line[4:8] ]

					groundtruths.append( self.tGT(cls_str, x1, y1, x2, y2, alpha, trunc, occ) )

		except:
			print '[GT] Cannot load %s.\n' % file_name

			import ipdb
			ipdb.set_trace()

		return groundtruths


	def _load_detections(self, file_name):		
		try:
			with open(file_name, 'r') as f:
				lines = [ line.rstrip('\n').split(' ') for line in f.readlines() ]
				detections = []
				for line in lines:
					cls_str = line[0]
					alpha = float(line[3])
					x1, y1, x2, y2 = [ float(item) for item in line[4:8] ]
					thr = float(line[-1])

					detections.append( self.tDT(cls_str, x1, y1, x2, y2, alpha, thr) )

		except:
			print '[DT] Cannot load %s.\n' % file_name
			
			import ipdb
			ipdb.set_trace()

		return detections


	class tPrData():
		__slots__ = ('v', 'similarity', 'tp', 'fp', 'fn')
		def __init__(self):
			self.v = []
			self.similarity = 0.0
			self.tp = 0
			self.fp = 0
			self.fn = 0

	# class tBox():
	# 	__slots__ = ('type', 'box', 'alpha')
	# 	def __init__(self, type, x1, y1, x2, y2, alpha):
	# 		self.type = type
	# 		self.box = [x1, y1, x2, y2, alpha]

	class tGT():
		__slots__ = ('type', 'box', 'truncation', 'occlusion')
		def __init__(self, type='invalid', x1=-1, y1=-1, x2=-1, y2=-1, alpha=-10, trunc=-1, occ=-1):
			# tBox.__init__(self, type, x1, y1, x2, y2, alpha)			
			self.type = type
			self.box = [x1, y1, x2, y2, alpha]
			self.truncation = trunc
			self.occlusion = occ


	class tDT():
		__slots__ = ('type', 'box', 'thresh')
		def __init__(self, type='invalid', x1=-1, y1=-1, x2=-1, y2=-1, alpha=-10, thr=-1000):
			# tBox.__init__(self, type, x1, y1, x2, y2, alpha)			
			self.type = type
			self.box = [x1, y1, x2, y2, alpha]
			self.thresh = thr

	# class tGT():
	# 	__slots__ = ('box', 'truncation', 'occlusion')
	# 	def __init__(self, box=tBox('invalid',-1,-1,-1,-1,-10), trunc=-1, occ=-1):
	# 		self.box = box
	# 		self.truncation = trunc
	# 		self.occlusion = occ
	# __slots__ = ('box', 'thresh')
	# def __init__(self, box=tBox('invalid',-1,-1,-1,-1,-10), thr=-1000):
	# 	self.box = box
	# 	self.thresh = thr