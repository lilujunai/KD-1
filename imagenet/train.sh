MODEL='efficientnet-b0'
TEACHER='resnet152'
BATCH_SIZE=256

w=0.4
GPUS=5,6,7,10,11
echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 4						\
        --T 3                                                   \
        --w $w                                                  \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.1                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}_${w}" --epochs 300

echo "first test done: $(date)"

w=0.6
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 4						\
        --T 3                                                   \
        --w $w                                                  \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.1                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}_${w}" --epochs 300

echo "second test done: $(date)"

w=0.8
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 4						\
        --T 3                                                   \
        --w $w                                                  \
        --teacher_arch $TEACHER                                 \
        --batch-size $BATCH_SIZE                                \
        --lr 0.1                                               \
        --kd                                                    \
        --overhaul                                              \
        --save_path "weights/${TEACHER}_${MODEL}_${w}" --epochs 300


echo "last test done: $date"
