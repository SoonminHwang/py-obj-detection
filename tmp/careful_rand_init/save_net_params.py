for key, blobvec in net.params.items():
	weight = blobvec[0].data.copy()
	try:
		n, c, w, h = weight.shape
		weight = weight.reshape((n*w, c*h))
	except:
		print( 'ndim: %d' % len(weight.shape) )
				
	weight = weight * 256 * 256
	
	key = key.replace('/', '_')
	skimage.io.imsave( key + '_int16.tif', weight.astype(np.int16) )
