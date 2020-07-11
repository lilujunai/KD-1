GPUS=12

CUDA_VISIBLE_DEVICES=$GPUS python3 main.py  \
	--arch efficientnet-b0  \
  	--batch_size 64
