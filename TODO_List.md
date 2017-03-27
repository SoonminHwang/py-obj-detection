### 17.03.15.
0. KITTI-Tracking
		
	    - make velo_dispmap (Object/Training, Tracking)
	    - make 3D video (PyQt) 
	        - with velo_dispmap
	        - with mc-cnn

0. Check Flow2015 Dense GT
		
		- Load & draw as point cloud
        - Among 200 frames in flow set, only 142 frames are from raw set.
        - Among the 142 frames, only 93 frames are element of object set.
        - 93 (object/training) = 46 (3dop, train) + 47 (3dop, val)

0. Read papers

		- Xception
	
0. Implement

	    - accuracy layer
	    - multi-scale training (crop trianing example)
	    - multi-scale cnn (variable receptive field)
	    - ignore handling
	    
0. Training
	    
		- [O] new depth2 ZF
	    - [O] new depth2 VGG16
	    - context2x pooling, new depth2 ZF
	    - Caltech / KAIST dataset
0. Etc.
    
		- copy velo_dispmap from 178 to 17 & 34
	        - [O] Object/training
	        - Object/testing
	        - Tracking/training
	        - Tracking/testing
	    - [Ready] run make velo_dispmap for tracking set on 178


### 17.03.16.
0. [O] Use png image for depth
0. [O] Multi-scale training (test, 34, rgb only): More iteration!
0. [O] All-in-one prototxt (+Solver)

		- Issue: training without validation mode

0. [Ready] Train a network from random init
 
    	Careful init (sigma of gaussian for each layer)

0. [Ongoing] Implement mAP layer
0. Try Inception-style network
0. Try Xception-residual network
0. Consider receptive fields


### 17.03.17.
0. [O] BugFix: Load All-in-one network in demo.py (deploy stage)
0. [O] Train a network from random init

    	Careful init (sigma of gaussian for each layer)

0. [Ongoing] Implement mAP layer
0. Debug: context pooling / pool2x (in mscnn_ext branch)
0. Add feature

		- try to load specified pre-trained network, if it failed, just use random initialization

### On Weekend,

0. Training

	    - Multi-scale / 
	        [ZF] RGB from voc init
	        [ZF] RGB+D (2) from random init
	        [ZF] RGB+D (2) from imagenet init
	        [VGG16] RGB from imagenet init (baseline)

0. Implement

		- PyQT interface: show & recode points clouds stream
		- Finish mAP layer


### 17.03.20.
0. Survey: Few-shot learning
1. [DEBUG] Finish mAP layer
2. [Ongoing] PyQT Interface


### 17.03.21.
0. Finish mAP layer
1. [O][Initial] PyQT Interface

### 17.03.22.
1. [Update] PyQT Interface

		- Upload to github repository
		- Added feature: load from binary files for fast loading.
		
2. [O] Plot performance variation [170222-11-25-58]
3. Read papers

		- Mask R-CNN
		- [O] Multi-view 3D Object Detection network for Autonomous Driving

4. Prepare a new network

		- Draw 3D bounding boxes
		- Make quantized 3D point clouds with RGB
		- Write prototxt

0. [O][Initial] Finish mAP layer


### 17.03.23.
0. Debug: depth input
1. Depth extension -> XYD input (metric?)
2. [O][Not helpful] Test: bigger image (train 2x, test 3x)
3. Research design

		- Velodyne point refinement
		- Depth hallucination
		- all around rgb-d detection

4. Try another backbone
 
		- ResNetXt

5. Another depth estimation algorithm

		- dispnet (stereo)
		- linear interpolation (poor than least square)
		- mc-cnn (100)