export PYTHONPATH=../:$PYTHONPATH


VIC_MODEL=vgg16
VIC_DATASET=CIFAR10

python knockoff/adversary/train.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
$VIC_MODEL $VIC_DATASET \
--budgets 500,1000,1500,2000,3000,4000,5000 \
-d $1 \
--pretrained imagenet_for_cifar \
--epochs 10 \
--log-interval 25 \
--lr 0.1 

