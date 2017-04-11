#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
EXP_NAME=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  kaistv2)
    DB_NAME="kaistv2"
    #TRAIN_IMDB="kaist_2015_train20"
    TRAIN_IMDB="kaistv2_2015_train01"
    TEST_IMDB="kaistv2_2015_test20"
    PT_DIR="kaistv2"
    CONFIG="faster_rcnn_end2end_kaistv2.yml"    
    #ITERS=150000
    ;;    
  kaist)
    DB_NAME="kaist"
    #TRAIN_IMDB="kaist_2015_train20"
    TRAIN_IMDB="kaist_2015_train01"
    TEST_IMDB="kaist_2015_test20"
    PT_DIR="kaist"
    CONFIG="faster_rcnn_end2end_kaist.yml"    
    #ITERS=150000
    ;;
  kitti_trainval)
    DB_NAME="kitti_all"
    TRAIN_IMDB="kitti_2012_trainval"
    TEST_IMDB="kitti_2012_val"
    PT_DIR="kitti"
    CONFIG="faster_rcnn_end2end_kitti_${NET}.yml"
    #ITERS=100000
    ;;
  kitti)
    DB_NAME="kitti"
    TRAIN_IMDB="kitti_2012_train"
    TEST_IMDB="kitti_2012_val"
    PT_DIR="kitti"
    CONFIG="faster_rcnn_end2end_kitti_${NET}.yml"
    #ITERS=450000   # For AlexNet
    #ITERS=20000
    #ITERS=150000
    ;;
  voc_0712)
    DB_NAME="voc0712"
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2012_test"
    PT_DIR="pascal_voc"
    CONFIG="faster_rcnn_end2end.yml"
    #ITERS=100000
    ;;
  pascal_voc)
    DB_NAME="voc07"
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    CONFIG="faster_rcnn_end2end_voc07_${NET}.yml"
    #ITERS=70000
    ;;
  coco14_trainval)
    DB_NAME="coco14_all"
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train+coco_2014_val"
    TEST_IMDB="coco_2015_test"
    PT_DIR="coco14_trainval"
    CONFIG="faster_rcnn_end2end.yml"
    #ITERS=70000
    ;;
  coco)
    DB_NAME="coco"
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    #TEST_IMDB="coco_2014_minival"
    TEST_IMDB="coco_2014_val"
    PT_DIR="coco"
    CONFIG="faster_rcnn_end2end.yml"
    #ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

DATE="`date +'%Y-%m-%d'`"
DST_DIR="experiments/logs/${DATE}/[`date +'%Y-%m-%d_%H-%M-%S'`][${NET}][${DB_NAME}]_${EXP_NAME}"
mkdir -p $DST_DIR
mkdir -p "${DST_DIR}/models"

LOG="${DST_DIR}/${DB_NAME}_${NET}_${EXTRA_ARGS_SLUG}.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo Copy .cfg "&" .prototxt "&" evaluation script files
cp "./tools/demo.py" $DST_DIR
#cp "./tools/eval_${DB_NAME}.py" $DST_DIR
#cp ./tools/kitti_evaluate_object.py $DST_DIR
#cp ./models/$PT_DIR/$NET/faster_rcnn_end2end/solver.prototxt "${DST_DIR}/models"
#cp ./models/$PT_DIR/$NET/faster_rcnn_end2end/trainval.prototxt "${DST_DIR}/models"
#cp ./models/$PT_DIR/$NET/faster_rcnn_end2end/test.prototxt "${DST_DIR}/models"
cp ./models/$PT_DIR/$NET.prototxt "${DST_DIR}/models/network.prototxt"
cp ./tools/testAll.py $DST_DIR
cp "experiments/cfgs/${CONFIG}" $DST_DIR


time ipython ./tools/train_net.py -- --gpu ${GPU_ID} \
  --imdb_train ${TRAIN_IMDB} \
  --imdb_val ${TEST_IMDB} \
  --cfg "experiments/cfgs/${CONFIG}" \
  --log_dir ${DST_DIR} \
  --rand \
  --weights data/imagenet_models/${NET}.v2.caffemodel
#  --set ${EXTRA_ARGS}
set +x
#NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
NET_FINAL=`grep -B 2 "done solving" ${LOG} | grep "Snapshotting to binary proto file" | awk '{print $10}'`
ITER_FINAL=`grep "experiments" ${NET_FINAL} | awk 'BEGIN {FS="/"}{print $6}' | awk 'BEGIN {FS="."}{print $1}'` 
set -x

echo Test a few images with final model
cd $DST_DIR
./demo.py --gpu ${GPU_ID} --iter ${ITER_FINAL} --imdb ${TEST_IMDB}
cd ../../../../

# time ./tools/test_net.py --gpu ${GPU_ID} \
#   --def "${DST_DIR}/models/test.prototxt" \
#   --net ${NET_FINAL} \
#   --imdb ${TEST_IMDB} \
#   --cfg "${DST_DIR}/${CONFIG}" \
#   --output_dir "${DST_DIR}/results/${ITER_FINAL}/"
#   ${EXTRA_ARGS}