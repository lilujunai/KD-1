GPUS=0,1
  
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py \
	--kd                    \
  --batch_size 64         \
  --lr 6e-4