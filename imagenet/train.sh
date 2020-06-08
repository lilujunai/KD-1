MODEL='efficientnet-b0'
TEACHER='resnet152'
BATCH_SIZE=150
w=0.8
GPUS=10,11,12,13,17
echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /home/vision/keti/data/Imagenet      \
        --arch $MODEL                                           \
        --workers 16						\
        --T 3                                                   \
        --w $w                                                  \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.01                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}_${w}" --epochs 300

echo "first test done: $(date)"

w=0.6
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py > kd_${TEACHER}_${w}.txt /home/vision/keti/data/Imagenet      \
        --arch $MODEL                                           \
        --workers 16						\
        --T 3                                                   \
        --w $w                                                  \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.01                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}_${w}" --epochs 300

echo "second test done: $(date)"

w=0.4
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py > kd_${TEACHER}_${w}.txt /home/vision/keti/data/Imagenet      \
        --arch $MODEL                                           \
        --workers 16						\
        --T 3                                                   \
        --w $w                                                  \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.01                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}_${w}" --epochs 300


echo "last test done: $date"
