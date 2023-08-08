export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
for VIC_MODEL in vgg16_bn
do
for VIC_DATASET in CIFAR10 CIFAR100
do
for MODE in  block_shallow block_deep
do


python knockoff/adversary/eval_stealing_graybox.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
$VIC_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 500,1000 \
-d $1 \
--graybox-mode $MODE \
&

done
done
done