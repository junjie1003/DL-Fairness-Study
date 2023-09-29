python boxplot_acc.py -d celeba -s gender -t blond_hair
python boxplot_acc.py -d utkface -s age -t gender
python boxplot_acc.py -d utkface -s race -t gender
python boxplot_acc.py -d cifar10s

python boxplot_acc_transpose.py -d celeba -s gender -t blond_hair
python boxplot_acc_transpose.py -d utkface -s age -t gender
python boxplot_acc_transpose.py -d utkface -s race -t gender
python boxplot_acc_transpose.py -d cifar10s
