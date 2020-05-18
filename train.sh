MODEL='efficientnet-b0'
TEACHER='resnext101_32x16d'
BATCH_SIZE=89
#272 _16

GPUS=0,1,2,3

CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /home/vision/keti/data/Imagenet	\
	--arch $MODEL						\
	--workers 55						\
	--T 3							\
	--w 0.6							\
	--teacher_arch $TEACHER					\
	--batch-size $BATCH_SIZE				\
	--kd							\
	--lr 0.000001						\
	--pretrained						\
	--save_path "weights/${MODEL}_kd" --epochs 10

