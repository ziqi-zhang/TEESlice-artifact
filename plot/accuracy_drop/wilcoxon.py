from scipy import stats

resnet18_baseline = [95.47, 79.94, 87.51, 86.97]
resnet18_teeslice = [93.65, 76.79, 86.22, 88.24]
resnet34_baseline = [91.11, 81, 88.22, 87.69]
resnet34_teeslice = [91.75, 76.53, 86.15, 89.55]
vgg16_baseline = [91.62, 73.03, 89.67, 89.19]
vgg16_teeslice = [93.06, 73.11, 89.42, 89.46]
vgg19_baseline = [92.48, 71.38, 89.62, 89.96]
vgg19_teeslice = [92.70, 73.15, 90.70, 89.46]

baseline = resnet18_baseline + resnet34_baseline + vgg16_baseline + vgg19_baseline
teeslice = resnet18_teeslice + resnet34_teeslice + vgg16_teeslice + vgg19_teeslice

res = stats.wilcoxon(baseline, teeslice, alternative="two-sided")
print(res.pvalue)
