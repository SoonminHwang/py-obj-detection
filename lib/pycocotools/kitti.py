__author__ = 'soonmin'
__version__ = '0.1'

# Interface for accessing the KITTI dataset.
#
# The KITTI class inherited from COCO class

from coco import COCO
import itertools

class KITTI(COCO):
	def __init__(self, annotation_file=None):
		"""
        Constructor of KITTI helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
		COCO.__init__(self, annotation_file)


	def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], occLevel=[], truncRng=[], alphaRng=[], hRng=[]):
		"""
		Get ann ids that satisfy given filter conditions. default skips that filter
		:param imgIds  (int array)     : get anns for given imgs
		:param catIds  (int array)     : get anns for given cats		
		:param areaRng (float array)   : get anns for given area range (e.g. [0 inf])
		:param occLevel (int array)    : get anns for given occ level
		:param truncRng (float array)  : get anns for given truncated range (e.g. [0 1])        
		:param alphaRng (float array)  : get anns for given angle range (e.g. [-pi..pi])                   
		:param hRng (float array)   : get anns for given height range (e.g. [0 inf])
		:return: ids (int array)       : integer array of ann ids
		"""
		imgIds = imgIds if type(imgIds) == list else [imgIds]
		catIds = catIds if type(catIds) == list else [catIds]

		if len(imgIds) == len(catIds) == len(areaRng) == len(occLevel) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(imgIds) == 0:
				lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
				anns = list(itertools.chain.from_iterable(lists))
			else:
				anns = self.dataset['annotations']

			anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]			
			anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] >= areaRng[0] and ann['area'] <= areaRng[1]]
			anns = anns if len(occLevel) == 0 else [ann for ann in anns if ann['occ'] in occLevel]
			anns = anns if len(truncRng) == 0 else [ann for ann in anns if ann['trunc'] >= truncRng[0] and ann['trunc'] <= truncRng[1]]
			anns = anns if len(alphaRng) == 0 else [ann for ann in anns if ann['alpha'] >= alphaRng[0] and ann['alpha'] <= alphaRng[1]]
			anns = anns if len(hRng) == 0 else [ann for ann in anns if ann['bbox'][3] >= hRng[0] and ann['bbox'][3] <= hRng[1]]

		ids = [ann['id'] for ann in anns]

		return ids

	def description(self):
		"""
		Print format description about the annotation file.
		:return:
		"""
		for key, value in self.dataset['description'].items():
			if key == 'occ':
				print '%s[%d]: %s'%(key, value['id'], value['desc'])
			else:
				print '%s: %s'%(key, value)