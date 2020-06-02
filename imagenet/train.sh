MODEL='efficientnet-b0'
TEACHER='resnet152'
BATCH_SIZE=400

GPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /home/vision/keti/data/Imagenet      \
        --arch $MODEL                                           \
        --workers 16						\
        --T 3                                                   \
        --w 0.8                                                 \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.01                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 300

