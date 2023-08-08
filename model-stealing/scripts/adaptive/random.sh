export PYTHONPATH=../:$PYTHONPATH

PROXY_MODEL=resnet18
for VIC_DATASET in CIFAR100
do
for VIC_MODEL in resnet18 resnet34 vgg16_bn vgg19_bn
do


python knockoff/adversary/transfer_nettailor.py random \
models_nettailor/models/$VIC_DATASET-$VIC_MODEL/$PROXY_MODEL-iterative-nettailor-3Skip-T10.0-C0.3-Pruned0 \
$VIC_DATASET \
--out_dir models_nettailor/adaptive_attack/full_model/victim[$VIC_DATASET-$VIC_MODEL]-proxy[$PROXY_MODEL]-random \
--budget 30000 \
--queryset $VIC_DATASET \
--batch_size 32 \
-d $1 \
--backbone $PROXY_MODEL \
--max-skip 3 \
--task $VIC_DATASET \
&


done
done