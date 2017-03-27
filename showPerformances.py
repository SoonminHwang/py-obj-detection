import numpy as np

iters = np.arange( 0, 500000, 10000 ) + 10000
#cls = 'Car'
classes = ['Car', 'Pedestrian', 'Cyclist']

rstFileFormat = 'experiments/logs/2017-03-24/[2017-03-24_03-10-07][VGG16_c4][kitti]_conv4_scale2x_PedCyc2x_200k1000k_person_sitting_van_truck_imdb_hRng10_occ012_trunc0/results/vgg16_iter_%d/plots/%s_detection.txt'
# rstFileFormat = 'experiments/logs/2017-03-23/[2017-03-23_04-17-52][VGG16_c4][kitti]_conv4_scale2x_PedCyc2x_50k120k_person_sitting_van_truck_imdb_hRng10_occ012_trunc0/results/vgg16_iter_%d/plots/%s_detection.txt'
#rstFileFormat = 'experiments/logs/2017-02-22/[2017-02-22_11-25-58][ZF][kitti]_scale2x_10x7_stepsize_100k_maxiter_140k_person_sitting_van_truck_imdb_hRng20_occ012_trunc0.5/results/zf_iter_%d/plots/%s_detection.txt'

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


cond = 0
car_moderate = [ r['Car'][cond] for r in results ]
ped_moderate = [ r['Pedestrian'][cond] for r in results ]
cyc_moderate = [ r['Cyclist'][cond] for r in results ]
iters = [ r['iter'] for r in results ]

import matplotlib.pyplot as plt
import seaborn as sns

clrs = sns.color_palette("Set2", 3)

plt.figure(1)
plt.plot( iters, car_moderate, color=clrs[0], label='Car' )
plt.plot( iters, ped_moderate, color=clrs[1], label='Pedestrian' )
plt.plot( iters, cyc_moderate, color=clrs[2], label='Cyclist' )

# plt.title('[2017-02-22_11-25-58][ZF][kitti]_scale2x_10x7_stepsize_100k_maxiter_140k')
# plt.title('[2017-03-23_04-17-52][VGG16_c4][kitti]_conv4')
plt.title('[2017-03-24_03-10-07][VGG16_c4][kitti]_conv4')

import ipdb
ipdb.set_trace()

# plt.gca().set_xticklabels(iters)

plt.gca().text(iters[-2], car_moderate[-1]-10, '{:.2f}'.format(car_moderate[-1]), 
            bbox=dict(facecolor=clrs[0], alpha=0.5), fontsize=14, color='white')
plt.gca().text(iters[-3], ped_moderate[-1]-5, '{:.2f}'.format(ped_moderate[-1]), 
            bbox=dict(facecolor=clrs[1], alpha=0.5), fontsize=14, color='white')
plt.gca().text(iters[-4], cyc_moderate[-1]-10, '{:.2f}'.format(cyc_moderate[-1]), 
            bbox=dict(facecolor=clrs[2], alpha=0.5), fontsize=14, color='white')

plt.legend(loc='best')

plt.savefig('conv4_perf_moderate.png', dpi=200)



plt.show()
