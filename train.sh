MODEL='efficientnet-b0'
TEACHER='resnet152'
BATCH_SIZE=54

GPUS=0,1

CUDA_VISIBLE_DEVICES=$GPUS python3 main.py > ${TEACHER}_${MODEL}.txt /home/taeil/research/data/Imagenet      \
        --arch $MODEL                                           \
        --workers 55                                            \
        --T 3                                                   \
        --w 0.6                                                 \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.000001                                           \
        --pretrained                                            \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 10

