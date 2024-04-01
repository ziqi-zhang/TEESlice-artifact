export PYTHONPATH=../:$PYTHONPATH

MODEL=alexnet
for MODEL in resnet34 vgg19_bn 
do
for DATASET in CIFAR10 STL10 UTKFaceRace 
do


python train_shadownet.py $DATASET $MODEL \
-d $1 \
-o results/adversary/shadownet/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
--shadow_model_dir results/adversary/shadow/$DATASET-$MODEL \
--budgets 50 \
--lr 0.1 \
--remain-lr 1e-3 \
--update-lr 1e-2 \
--epochs 10 \
--pretrained \
--argmaxed &


done
done