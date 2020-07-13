MODEL='efficientnet-b0'

GPUS=0,1,2,3,4

echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 300

echo "test done: $(date)"
