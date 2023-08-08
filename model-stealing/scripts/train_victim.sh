export PYTHONPATH=../:$PYTHONPATH

MODEL=resnet50
for DATASET in CIFAR100 CIFAR10
do

python knockoff/victim/train.py $DATASET $MODEL \
-d $1 \
-o models/victim/$DATASET-$MODEL \
-e 20 \
--log-interval 25 \
--pretrained imagenet_for_cifar \
--lr 0.1 \
--lr-step 10 &



done