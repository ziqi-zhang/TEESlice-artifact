export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
for VIC_MODEL in vgg16_bn vgg19_bn
do
for VIC_DATASET in CIFAR100 
do
for MODE in block_large_mag
do


python knockoff/adversary/train_mag.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
$VIC_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 100,300 \
-d $1 \
--pretrained imagenet_for_cifar \
--log-interval 100 \
--epochs 10 \
--lr 0.01 \
--graybox-mode $MODE \
--argmaxed \
&



done
done
done