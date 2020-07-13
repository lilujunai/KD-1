GPUS=14
  
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py \
  --kd                    \
  --batch_size 64         \
  --lr 6e-4		\
  --student_path './checkpoint/EfficientNet:92.02.pth'\
  --teacher_path './checkpoint/ResNet:93.80.pth'
