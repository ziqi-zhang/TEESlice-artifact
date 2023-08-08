import os, sys
# sys.path.append('models/')
from pdb import set_trace as st

ROOT = "models_nettailor/models"
VICTIM_ROOT = "models_face/victim"

# Supported backbone CNNs: 'resnet18', 'resnet34', 'resnet50'
BACKBONE = 'resnet18'

# Select GPU id
GPU = 0
PRETRAIN=0

BACKBONE = 'resnet18'
for VICTIM_ARCH in [
	# 'resnet18', 
	# 'vgg16_bn'
	'alexnet'
]:

	for TASK in [
		"STL10", 
		# "CIFAR100"
	]:

		########################## TRAIN target ############################################### 
		VICTIM_PATH = f"{VICTIM_ROOT}/{TASK}-{VICTIM_ARCH}"
		BATCH_SIZE = 64
		EPOCHS = 20
		LR = 0.01
		LR_EPOCHS = 20
		WEIGHT_DECAY = 0.0004

		target_dir = f'{ROOT}/{TASK}-{VICTIM_ARCH}/{BACKBONE}'

		cmd = (f"CUDA_VISIBLE_DEVICES={GPU} "
			f"python knockoff/nettailor/distill_teacher.py "
			f"--task {TASK} "
			f"--model-dir {target_dir} "
			f"--arch {BACKBONE} "
			f"--epochs {EPOCHS} "
			f"--batch-size {BATCH_SIZE} "
			f"--lr {LR} "
			f"--lr-decay-epochs {LR_EPOCHS} "
			f"--weight-decay {WEIGHT_DECAY} "
			f"--workers 4 "
			f"--imagenet_pretrain "
			f"--momentum 0.5 "
			f"--victim-arch {VICTIM_ARCH} "
			f"--victim-path {VICTIM_PATH} "
			f"--log2file ")

		print(cmd)
		os.system(cmd)

		print(cmd + ' --evaluate')
		os.system(cmd + ' --evaluate')
	
		######################### TRAIN Teacher ############################################### 
	
		target_fn = target_dir + '/checkpoint.pth.tar'

		# ######################### TRAIN STUDENT ############################################### 

		COMPLEXITY_COEFF = 0.3
		TEACHER_COEFF = 10.0
		MAX_SKIP = 3

		BATCH_SIZE = 64
		EPOCHS = 40
		LR_EPOCHS = 30
		LR = 0.1

		full_model_fn = f'{ROOT}/{TASK}-{VICTIM_ARCH}/{BACKBONE}-nettailor-{MAX_SKIP}Skip-T{TEACHER_COEFF}-C{COMPLEXITY_COEFF}'

		cmd = (f"CUDA_VISIBLE_DEVICES={GPU} "
			f"python knockoff/nettailor/nettailor_student.py "
			f"--task {TASK} "
			f"--model-dir {full_model_fn} "
			f"--teacher-fn {target_fn} "
			f"--backbone {BACKBONE} "
			f"--max-skip {MAX_SKIP} "
			f"--complexity-coeff {COMPLEXITY_COEFF} "
			f"--teacher-coeff {TEACHER_COEFF} "
			f"--epochs {EPOCHS} "
			f"--batch-size {BATCH_SIZE} "
			f"--lr {LR} "
			f"--lr-decay-epochs {LR_EPOCHS} "
			f"--weight-decay {WEIGHT_DECAY} "
			f"--workers 4 "
			f"--eval-freq 2 "
			f"--log2file ")

		print(cmd)
		os.system(cmd)

		print(cmd + " --evaluate")
		os.system(cmd + " --evaluate")

		# # # ########################## My Iterative Pruning ############################################### 

		NUM_BLOCKS_PRUNED = 0
		BATCH_SIZE = 64
		EPOCHS = 20
		LR = 0.01
		LR_EPOCHS = 20
	
		TOLERANCE = 0.01
		ITERATIVE_RATIO = 0.05
		PRUNE_INTERVAL = 2
		PROXY_PRUNING_THRESHOLD = 0.1
		PROXY_PRUNING_PERCENT = 0.5

		# EPOCHS = 2
		# synthesize_round_per_epoch = 1

		pruned_model_fn = f'{ROOT}/{TASK}-{VICTIM_ARCH}/{BACKBONE}-iterative-nettailor-{MAX_SKIP}Skip-T{TEACHER_COEFF}-C{COMPLEXITY_COEFF}-Pruned{NUM_BLOCKS_PRUNED}'
		cmd = (f"CUDA_VISIBLE_DEVICES={GPU} "
			f"python3 knockoff/nettailor/nettailor_iterative.py "
			f"--task {TASK} "
			f"--model-dir {pruned_model_fn} "
			f"--full-model-dir {full_model_fn} "
			f"--n-pruning-universal {NUM_BLOCKS_PRUNED} "
			f"--thr-pruning-proxy {PROXY_PRUNING_THRESHOLD} "
			f"--teacher-fn {target_fn} "
			f"--backbone {BACKBONE} "
			f"--max-skip {MAX_SKIP} "
			f"--complexity-coeff 0.0 "
			f"--teacher-coeff {TEACHER_COEFF} "
			f"--epochs {EPOCHS} "
			f"--batch-size {BATCH_SIZE} "
			f"--lr {LR} "
			f"--lr-decay-epochs {LR_EPOCHS} "
			f"--workers 4 "
			f"--eval-freq 2 "
			f"--start-prune-percent {PROXY_PRUNING_PERCENT} "
			f"--iterative-prune-ratio {ITERATIVE_RATIO} "
			f"--prune-interval {PRUNE_INTERVAL} "
			f"--tolerance {TOLERANCE} "
			f"--log2file ")
			# .format(
			# 		gpu=GPU, task=TASK, pruned_model_fn=pruned_model_fn, full_model_fn=full_model_fn, backbone=BACKBONE, 
			# 		teacher_fn=target_fn, 
			# 		thr=NUM_BLOCKS_PRUNED, adapt_thr=PROXY_PRUNING_THRESHOLD, max_skip=MAX_SKIP, teacher=TEACHER_COEFF, epochs=EPOCHS, 
			# 		bs=BATCH_SIZE, lr=LR, lr_epochs=LR_EPOCHS, 
			# 		start_percent=PROXY_PRUNING_PERCENT, tolerance=TOLERANCE, iterative_ratio=ITERATIVE_RATIO, interval=PRUNE_INTERVAL)

		print(cmd)
		os.system(cmd)

		print(cmd + " --evaluate")
		os.system(cmd + " --evaluate")