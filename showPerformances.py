import numpy as np

iters = np.arange( 0, 140000, 2000 ) + 2000
#cls = 'Car'
classes = ['Car', 'Pedestrian', 'Cyclist']

rstFileFormat = 'experiments/logs/2017-02-22/[2017-02-22_11-25-58][ZF][kitti]_scale2x_10x7_stepsize_100k_maxiter_140k_person_sitting_van_truck_imdb_hRng20_occ012_trunc0.5/results/zf_iter_%d/plots/%s_detection.txt'

results = []
for iter in iters:
	rDict = {}
	try:
		for cls in classes:
			rstFile = rstFileFormat % (iter, cls)

			with open( rstFile, 'rt' ) as f:
				rst = np.vstack( [ np.array( [ float(num) for num in line.rstrip('\n').split(' ') ] ) for line in f.readlines() ] )

				rDict[cls] = np.mean( rst[:,1:], axis=0 ) * 100

		rDict['iter'] = iter
		results.append( rDict )
	except:
		print 'Cannot load results from %s' % rstFile



car_moderate = [ r['Car'][0] for r in results ]
ped_moderate = [ r['Pedestrian'][0] for r in results ]
cyc_moderate = [ r['Cyclist'][0] for r in results ]

import matplotlib.pyplot as plt
import seaborn as sns

clrs = sns.color_palette("Set2", 3)

plt.figure(1)
plt.plot( car_moderate, color=clrs[0], label='Car' )
plt.plot( ped_moderate, color=clrs[1], label='Pedestrian' )
plt.plot( cyc_moderate, color=clrs[2], label='Cyclist' )

plt.title('[2017-02-22_11-25-58][ZF][kitti]_scale2x_10x7_stepsize_100k_maxiter_140k')


plt.gca().text(len(car_moderate)*0.9, car_moderate[-1]-10, '{:.2f}'.format(car_moderate[-1]), 
            bbox=dict(facecolor=clrs[0], alpha=0.5), fontsize=14, color='white')
plt.gca().text(len(ped_moderate)*0.9, ped_moderate[-1]-10, '{:.2f}'.format(ped_moderate[-1]), 
            bbox=dict(facecolor=clrs[1], alpha=0.5), fontsize=14, color='white')
plt.gca().text(len(cyc_moderate)*0.9, cyc_moderate[-1]-10, '{:.2f}'.format(cyc_moderate[-1]), 
            bbox=dict(facecolor=clrs[2], alpha=0.5), fontsize=14, color='white')

plt.legend(loc='best')

plt.savefig('perf.png', dpi=200)

plt.show()