export PYTHONPATH=../:$PYTHONPATH


for MODEL in  vgg19_bn
do
for DATASET in STL10  UTKFaceRace
# for DATASET in CIFAR10
do
for THETA in 1.0 0.9 0.7 0.5 0.3 0.2 0.1 0.0
# for THETA in 1.0
do


python train_soter.py $DATASET $MODEL \
-d $1 \
-o results/adversary/soter/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
--shadow_model_dir results/adversary/shadow/$DATASET-$MODEL \
--budgets 50 \
--lr 0.001 \
--epochs 10 \
--pretrained \
--argmaxed \
--soter_theta $THETA 


done
done
done