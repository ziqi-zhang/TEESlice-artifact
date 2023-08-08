export PYTHONPATH=../:$PYTHONPATH

PROXY_MODEL=resnet18


for VIC_MODEL in resnet18 resnet34 vgg16_bn vgg19_bn
do
for VIC_DATASET in CIFAR100
do


python knockoff/adversary_adaptive/train_full_model.py \
models_nettailor/adaptive_attack/full_model/victim[$VIC_DATASET-$VIC_MODEL]-proxy[$PROXY_MODEL]-random \
$VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
models_nettailor/models/$VIC_DATASET-$VIC_MODEL/$PROXY_MODEL-iterative-nettailor-3Skip-T10.0-C0.3-Pruned0 \
--budgets 50,100,300,500,1000,3000,5000,10000,15000,20000,25000,30000 \
-d $1 \
--pretrained imagenet_for_cifar \
--log-interval 100 \
--epochs 10 \
--lr 0.1 \
--backbone-lr 0.01 \
--argmaxed \
&


done
done