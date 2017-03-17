17.03.15.
KITTI-Tracking, 
    make velo_dispmap (Object/Training, Tracking)
    make 3D video (PyQt) 
        with velo_dispmap
        with mc-cnn
Check Flow2015 Dense GT
    [O] Load & draw as point cloud
        -> Among 200 frames in flow set, only 142 frames are from raw set.
        -> Among the 142 frames, only 93 frames are element of object set.
        -> 93 (object/training) = 46 (3dop, train) + 47 (3dop, val)
Read papers
Implement
    accuracy layer
    multi-scale training (crop trianing example)
    multi-scale cnn (variable receptive field)
    ignore handling
Training
    [O] new depth2 ZF
    [O] new depth2 VGG16
    context2x pooling, new depth2 ZF
    Caltech / KAIST dataset
Etc.
    copy velo_dispmap from 178 to 17 & 34
        [O] Object/training
        Object/testing
        Tracking/training
        Tracking/testing
    [Ready] run make velo_dispmap for tracking set on 178

17.03.16.
[O] Use png image for depth
[O] Multi-scale training (test, 34, rgb only): More iteration!
[O] All-in-one prototxt (+Solver)
    Issue: training without validation mode
[Ready] Train a network from random init
    Careful init (sigma of gaussian for each layer)
[Ongoing] Implement mAP layer
Try Inception-style network
Try Xception-residual network
Consider receptive fields

17.03.17.
[O] BugFix: Load All-in-one network in demo.py (deploy stage)
[O] Train a network from random init
    Careful init (sigma of gaussian for each layer)
[Ongoing] Implement mAP layer
Debug: context pooling / pool2x (in mscnn_ext branch)
Add feature: try to load specified pre-trained network, if it failed, just use random initialization

On Weekend,
    Multi-scale / 
        [ZF] RGB from voc init
        [ZF] RGB+D (2) from random init
        [ZF] RGB+D (2) from imagenet init
        [VGG16] RGB from imagenet init (baseline)
