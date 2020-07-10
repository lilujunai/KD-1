MODEL='efficientnet-b0'
TEACHER='resnet152'
BATCH_SIZE=512
GPUS=5,6,7,10,11,12

echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 8						\
        --T 3                                                   \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 6e-4                                               \
        --kd                                                    \
#        --pretrained                                            \
#        --pth_path  "./weights/resnet152_efficientnet-b0/model_best:EfficientNet_ResNet.pth.tar" \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 300

echo "test done: $(date)"
