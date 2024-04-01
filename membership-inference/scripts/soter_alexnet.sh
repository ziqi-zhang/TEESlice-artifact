export PYTHONPATH=../:$PYTHONPATH


for MODEL in  alexnet
do
# for DATASET in STL10 CIFAR10 CIFAR100 UTKFaceRace
for DATASET in CIFAR100
do
for THETA in 1.0 0.9 0.7 0.5 0.4 0.3 0.0
do


python train_soter.py $DATASET $MODEL \
-d $1 \
-o results/adversary/soter/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
--shadow_model_dir results/adversary/shadow/$DATASET-$MODEL \
--budgets 100 \
--lr 0.001 \
--epochs 10 \
--pretrained \
--argmaxed \
--soter_theta $THETA 


done
done
done