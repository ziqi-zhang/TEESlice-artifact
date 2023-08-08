
Goten/nettailor文件夹下的代码功能

- resnet.py: pytorch原始的resnet代码，对于输入图片是224x224
- resnet_small.py: 针对cifar的resnet代码，输入图片是32x32
- occlum_resnet.py: 复制到occlum环境中跑resnet的代码
- sgx_resnet.py: resnet的sgx的模型代码，输入图片是224x224
- sgx_resnet_small.py: 针对cifar的resnet的sgx的模型，输入图片是32x32
- sgx_student_resnet_small_full.py: 把nettailor模型转变成sgx的模型，这里nettailor的模型是裁剪之前的
- sgx_student_resnet_small_tailor.py: 把nettailor模型转变成sgx的模型，这里nettailor的模型可以是裁剪后/裁剪前都可以
- student_resnet_small.py: nettailor的student模型
- sgx_student_resnet_small_tailor_freivalds.py: 

文件夹
- model_export: 把nettailor模型导出成cfg文件
- cifar10: cifar10训好的模型
- resnet_baseline: 原本resnet模型运行时间的汇总
- scripts: 
    - scripts/run_resnet_baseline.sh: 运行sgx_student_resnet_small_tailor.py的脚本，可以把所有resnet模型都单独跑一遍记录时间，分别保存
    - scripts/collect_resnet_baseline.py: 把保存的脚本进行整理汇总