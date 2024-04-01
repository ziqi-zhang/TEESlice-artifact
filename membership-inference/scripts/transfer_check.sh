export PYTHONPATH=../:$PYTHONPATH

MODEL=resnet18
for DATASET in STL10
do

python transferset_check.py $DATASET $MODEL \
-d $1 \
-o results/adversary/train_layer/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
--trasnferset_budgets 1000 \
--budgets 500 \
--argmaxed


done