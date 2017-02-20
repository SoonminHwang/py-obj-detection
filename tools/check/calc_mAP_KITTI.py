import numpy as np
import os

def calculate_mAP(clsNm, pth='.'):

	recalls = []
	precisions = {'Easy':[], 'Moderate':[], 'Hard':[]}

	fName = os.path.join(pth, '{:s}_detection.txt'.format(clsNm))
	
	with open(fName, 'r') as fp:
		#lines = fp.readlines()
		lines = [line.rstrip().split(' ') for line in fp.readlines()]
		
		#assert len(lines[0]) == 4, '# of columns should be equal to 4.'

		for line in lines:
			recalls.append( float(line[0]) )
			precisions['Easy'].append( float(line[1]) )
			precisions['Moderate'].append( float(line[2]) )
			precisions['Hard'].append( float(line[3]) )	
	
	mAP = {'Easy': np.mean(precisions['Easy'][::4]), 
		   'Moderate': np.mean(precisions['Moderate'][::4]), 
		   'Hard': np.mean(precisions['Hard'][::4])}

	return recalls, mAP


