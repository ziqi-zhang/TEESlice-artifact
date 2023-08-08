export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
for VIC_MODEL in alexnet resnet18 resnet34 vgg19_bn vgg16_bn
do
for VIC_DATASET in CIFAR100 
do
for MODE in block_shallow
do


python knockoff/adversary/train_layers.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
$VIC_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 100,300 \
-d $1 \
--pretrained imagenet_for_cifar \
--log-interval 100 \
--epochs 10 \
--lr 0.1 \
--remain-lr 1e-3 \
--update-lr 1e-2 \
--graybox-mode $MODE \
--argmaxed \
&

done
done
done