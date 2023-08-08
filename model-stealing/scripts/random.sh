export PYTHONPATH=../:$PYTHONPATH


VIC_DATASET=CIFAR10
for VIC_MODEL in vgg19_bn
do
for VIC_DATASET in CIFAR100
do

python knockoff/adversary/transfer.py random \
models/victim/$VIC_DATASET-$VIC_MODEL \
--out_dir models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random \
--budget 5000 \
--queryset $VIC_DATASET \
--batch_size 32 \
-d $1 &

done
done