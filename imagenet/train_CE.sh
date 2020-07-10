MODEL='efficientnet-b0'
BATCH_SIZE=512
GPUS=0,1,2,3,4

echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 8				                                    	\
        --T 3                                                   \
        --batch-size $BATCH_SIZE                                \
        --lr 6e-4                                               \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 300

echo "test done: $(date)"
