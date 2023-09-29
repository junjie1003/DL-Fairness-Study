# DL-Fairness-Study

This code repository contains all the data processing and code implementations for our work titled "A Large-scale Empirical Study on Improving the Fairness of Deep Learning Models." Our research focuses on fairness issues in deep learning and includes a comprehensive summary and experimental analysis of existing fairness-improving methods on image data. We have also presented some interesting findings. In the future, we will continue to explore effective ways to enhance model fairness on image data.

## Experimental Environment

### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html) is sufficient if you do not already have conda installed.

Then, to create a new Python 3.8 environment, run:

```bash
conda create -n fairness python=3.8
conda activate fairness
```

### PyTorch

We have standardized the experimental environment for all methods, conducting experiments based on PyTorch. The installed versions and methods are as follows:

```bash
# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Requirements

Other necessary libraries and packages are installed as follows:

```bash
pip install -r requirements.txt
```

## Studied Methods in This Study

### Pre-processing

#### Undersampling (US)

```bash
# CelebA
python train_celeba/train_celeba_us.py --task blonde --epochs 170 --lr 1e-4 --gpu 0

# UTKFace
python train_utk_face/train_utk_face_us.py --task age --epochs 400 --epochs_extra 400 --lr 1e-4 --gpu 3
python train_utk_face/train_utk_face_us.py --task race --epochs 120 --epochs_extra 120 --lr 1e-4 --gpu 1

# CIFAR-10S
python train_cifar10/train_cifar10_us.py --epochs 2000 --lr 0.1 --gpu 0
```

#### Oversampling (OS)

```bash
# CelebA
python train_celeba/train_celeba_os.py --task blonde --epochs 4 --lr 1e-4 --gpu 0

# UTKFace
python train_utk_face/train_utk_face_os.py --task age --epochs 7 --lr 1e-4 --gpu 0
python train_utk_face/train_utk_face_os.py --task race --epochs 10 --lr 1e-4 --gpu 1

# CIFAR-10S
python train_cifar10/train_cifar10_os.py --epochs 100 --lr 0.1 --gpu 0
```

#### Upweighting (UW)

```bash
# CelebA
python train_celeba/train_celeba_uw.py --task blonde --epochs 10 --lr 1e-4 --gpu 0

# UTKFace
python train_utk_face/train_utk_face_uw.py --task age --epochs 20 --lr 1e-4 --gpu 0
python train_utk_face/train_utk_face_uw.py --task race --epochs 20 --lr 1e-4 --gpu 1

# CIFAR-10S
python train_cifar10/train_cifar10_uw.py --epochs 200 --lr 0.01 --gpu 0
```

#### Bias Mimicking (BM)

```bash
# CelebA
python train_celeba/train_celeba_bm.py --task blonde --epochs 10 --lr 1e-4 --lr_layer 1e-4 --gpu 0 --mode none
python train_celeba/train_celeba_bm.py --task blonde --epochs 10 --lr 1e-4 --lr_layer 1e-4 --gpu 1 --mode us
python train_celeba/train_celeba_bm.py --task blonde --epochs 10 --lr 1e-4 --lr_layer 1e-4 --gpu 2 --mode os
python train_celeba/train_celeba_bm.py --task blonde --epochs 10 --lr 1e-4 --lr_layer 1e-4 --gpu 3 --mode uw

# UTKFace
python train_utk_face/train_utk_face_bm.py --task age --epochs 20 --lr 1e-4 --gpu 0 --mode none
python train_utk_face/train_utk_face_bm.py --task race --epochs 20 --lr 1e-4 --gpu 1 --mode none

python train_utk_face/train_utk_face_bm.py --task age --epochs 20 --lr 1e-4 --gpu 0 --mode us
python train_utk_face/train_utk_face_bm.py --task race --epochs 20 --lr 1e-4 --gpu 1 --mode us

python train_utk_face/train_utk_face_bm.py --task age --epochs 20 --lr 1e-4 --gpu 0 --mode os
python train_utk_face/train_utk_face_bm.py --task race --epochs 20 --lr 1e-4 --gpu 1 --mode os

python train_utk_face/train_utk_face_bm.py --task age --epochs 20 --lr 1e-4 --gpu 0 --mode uw
python train_utk_face/train_utk_face_bm.py --task race --epochs 20 --lr 1e-4 --gpu 1 --mode uw

# CIFAR-10S
python train_cifar10/train_cifar10_bm.py --epochs 200 --lr 0.1 --gpu 0 --mode none
python train_cifar10/train_cifar10_bm.py --epochs 200 --lr 0.1 --gpu 1 --mode us
python train_cifar10/train_cifar10_bm.py --epochs 200 --lr 0.1 --gpu 2 --mode os
python train_cifar10/train_cifar10_bm.py --epochs 200 --lr 0.1 --gpu 3 --mode uw
```

### In-processing

#### Adversarial Training (Adv)

```bash
# CelebA
python train_celeba/train_celeba_adv.py --task blonde --epochs 10 --lr 1e-4 --gpu 0

# UTKFace
python train_utk_face/train_utk_face_adv.py --task age --epochs 20 --lr 1e-4 --gpu 0
python train_utk_face/train_utk_face_adv.py --task race --epochs 20 --lr 1e-4 --gpu 1

# CIFAR-10S
python train_cifar10/train_cifar10_adv.py --epochs 200 --lr 0.01 --gpu 0
```

#### Domain Independent Training (DI)

```bash
# CelebA
python train_celeba/train_celeba_di.py --task blonde --epochs 10 --lr 1e-4 --gpu 0

# UTKFace
python train_utk_face/train_utk_face_di.py --task age --epochs 20 --lr 1e-4 --gpu 0
python train_utk_face/train_utk_face_di.py --task race --epochs 20 --lr 1e-4 --gpu 1

# CIFAR-10S
python train_cifar10/train_cifar10_di.py --epochs 200 --lr 0.1 --gpu 0
```

#### Bias-Contrastive and Bias-Balanced Learning (BC+BB)

```bash
# CelebA
python train_celeba/train_celeba_bc.py --task blonde --epochs 10 --lr 1e-4 --gpu 0

# UTKFace
python train_utk_face/train_utk_face_bc.py --task age --epochs 20 --lr 1e-4 --gpu 0
python train_utk_face/train_utk_face_bc.py --task race --epochs 20 --lr 1e-4 --gpu 1

# CIFAR-10S
python train_cifar10/train_cifar10_bc.py --epochs 200 --lr 0.1 --gpu 0
```

#### FLAC

```bash
# CelebA
python train_celeba.py --task blonde --alpha 30000 --gpu 0

# UTKFace
python train_utk_face.py --task age --gpu 0
python train_utk_face.py --task race --gpu 1

# CIFAR-10S
python train_cifar10s.py --gpu 0
```

#### MMD-based Fair Distillation (MFD)

```bash
# CelebA
# 1. Train a teacher model
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --method scratch --dataset celeba --img-size 176 --repeat-time 1 > log/teacher_celeba_Blond_Hair.log

# 2. Train a student model
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --method kd_mfd --dataset celeba --labelwise --lambf 7 --lambh 0 --no-annealing --img-size 176 --teacher-path trained_models/230903/celeba/scratch/resnet_seed1_epochs50_bs128_lr0.001_Blond_Hair.pt > log/student_celeba_Blond_Hair.log

# 3. Evaluate student model
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --mode eval --method kd_mfd --dataset celeba --labelwise --lambf 7 --lambh 0 --no-annealing --img-size 176 --model-path trained_models/230903/celeba/kd_mfd/resnet_seed1_epochs50_bs128_lr0.001_rbf_sigma1.0_labelwise_temp3_lambh0.0_lambf7.0_fixedlamb_Blond_Hair.pt > log/eval_celeba_Blond_Hair.log

# UTKFace
# 1. Train a teacher model
CUDA_VISIBLE_DEVICES=1 python3 ./main.py --method scratch --dataset utkface --sensitive age --img-size 176 --pretrained --repeat-time 1 > log/teacher_utkface_age.log
CUDA_VISIBLE_DEVICES=2 python3 ./main.py --method scratch --dataset utkface --sensitive race --img-size 176 --pretrained --repeat-time 1 > log/teacher_utkface_race.log

# 2. Train a student model
CUDA_VISIBLE_DEVICES=1 python3 ./main.py --method kd_mfd --dataset utkface --sensitive age --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --teacher-path trained_models/230929/utkface/scratch/resnet_pretrained_seed1_epochs100_bs128_lr0.001_age.pt > log/student_utkface_age.log
CUDA_VISIBLE_DEVICES=2 python3 ./main.py --method kd_mfd --dataset utkface --sensitive race --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --teacher-path trained_models/230929/utkface/scratch/resnet_pretrained_seed1_epochs100_bs128_lr0.001_race.pt > log/student_utkface_race.log

# 3. Evaluate student model
CUDA_VISIBLE_DEVICES=1 python3 ./main.py --mode eval --method kd_mfd --dataset utkface --sensitive age --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --model-path trained_models/230929/utkface/kd_mfd/resnet_seed1_epochs100_bs128_lr0.001_rbf_sigma1.0_labelwise_temp3_lambh0.0_lambf3.0_fixedlamb_age.pt > log/eval_utkface_age.log
CUDA_VISIBLE_DEVICES=2 python3 ./main.py --mode eval --method kd_mfd --dataset utkface --sensitive race --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --model-path trained_models/230929/utkface/kd_mfd/resnet_seed1_epochs100_bs128_lr0.001_rbf_sigma1.0_labelwise_temp3_lambh0.0_lambf3.0_fixedlamb_race.pt > log/eval_utkface_race.log

# CIFAR-10S
# 1. Train a teacher model
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --method scratch --dataset cifar10s --img-size 32 --pretrained --repeat-time 1 > log/teacher_cifar10s.log

# 2. Train a student model
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --method kd_mfd --dataset cifar10s --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 32 --teacher-path trained_models/230929/cifar10s/scratch/resnet_pretrained_seed1_epochs100_bs128_lr0.001.pt > log/student_cifar10s.log

# 3. Evaluate student model
CUDA_VISIBLE_DEVICES=0 python3 ./main.py --mode eval --method kd_mfd --dataset cifar10s --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 32 --model-path trained_models/230929/cifar10s/kd_mfd/resnet_seed1_epochs100_bs128_lr0.001_rbf_sigma1.0_labelwise_temp3_lambh0.0_lambf3.0_fixedlamb.pt > log/eval_cifar10s.log
```

#### FDR

```bash
# CelebA
CUDA_VISIBLE_DEVICES=0 python celeba.py --ft_lr 1e-3 --ft_epoch 1000 --alpha 2 --constraint EO > log/CelebA_BlondHair_EO.log
CUDA_VISIBLE_DEVICES=1 python celeba.py --ft_lr 1e-3 --ft_epoch 500 --alpha 5 --constraint AE > log/CelebA_BlondHair_AE.log
CUDA_VISIBLE_DEVICES=2 python celeba.py --ft_lr 1e-3 --ft_epoch 1000 --constraint MMF > log/CelebA_BlondHair_MMF.log

# UTKFace
CUDA_VISIBLE_DEVICES=1 python utkface.py --ft_lr 1e-3 --ft_epoch 1500 --alpha 2 --constraint EO --sensitive age > log/UTKFace_Age_EO.log
CUDA_VISIBLE_DEVICES=2 python utkface.py --ft_lr 1e-3 --ft_epoch 1500 --alpha 2 --constraint EO --sensitive race > log/UTKFace_Race_EO.log

CUDA_VISIBLE_DEVICES=0 python utkface.py --ft_lr 3e-3 --ft_epoch 1500 --alpha 5 --constraint AE --sensitive age > log/UTKFace_Age_AE.log
CUDA_VISIBLE_DEVICES=1 python utkface.py --ft_lr 3e-3 --ft_epoch 1500 --alpha 5 --constraint AE --sensitive race > log/UTKFace_Race_AE.log

CUDA_VISIBLE_DEVICES=2 python utkface.py --ft_lr 1e-3 --ft_epoch 1000 --constraint MMF --sensitive age > log/UTKFace_Age_MMF.log
CUDA_VISIBLE_DEVICES=3 python utkface.py --ft_lr 1e-3 --ft_epoch 1000 --constraint MMF --sensitive race > log/UTKFace_Race_MMF.log

# CIFAR-10S
CUDA_VISIBLE_DEVICES=3 python cifar10s.py --ft_lr 1e-3 --ft_epoch 1000 --alpha 2 --constraint EO > log/CIFAR-10S_EO.log
CUDA_VISIBLE_DEVICES=3 python cifar10s.py --ft_lr 1e-3 --ft_epoch 1000 --alpha 5 --constraint AE > log/CIFAR-10S_AE.log
CUDA_VISIBLE_DEVICES=3 python cifar10s.py --ft_lr 1e-3 --ft_epoch 1000 --constraint MMF > log/CIFAR-10S_MMF.log
```

### Post-processing

#### FairReprogram (FR)

Step 1: Standard Training

```bash
# CelebA
nohup python3 train.py --dataset celeba --method std --result-dir std --gpu 0 > log/celeba_std.log 2>&1 &
# UTKFace
nohup python3 train.py --dataset utkface --domain-attrs Age --method std --result-dir std --gpu 1 > log/utkface_age_std.log 2>&1 &
nohup python3 train.py --dataset utkface --domain-attrs Race --method std --result-dir std --gpu 2 > log/utkface_race_std.log 2>&1 &
# CIFAR-10S
nohup python3 train.py --dataset cifar10s --method std --result-dir std --resume --gpu 3 > log/cifar10s_std.log 2>&1 &
```

Step 2: Reprogramming with Border Trigger

```bash
# CelebA
nohup python3 train.py --dataset celeba --result-dir border --reprogram-size 184 --epochs 20 --adversary-with-logits --lmbda 10.0 --m repro --adversary-with-y --checkpoint std/std_resnet18_celeba_seed1_best.pth.tar --gpu 0 > log/celeba_border_eo.log 2>&1 &
# UTKFace
nohup python3 train.py --dataset utkface --domain-attrs Age --result-dir border --reprogram-size 184 --epochs 20 --adversary-with-logits --lmbda 10.0 --m repro --adversary-with-y --checkpoint std/std_resnet18_utkface_age_seed1_best.pth.tar --gpu 1 > log/utkface_age_border_eo.log 2>&1 &
nohup python3 train.py --dataset utkface --domain-attrs Race --result-dir border --reprogram-size 184 --epochs 20 --adversary-with-logits --lmbda 10.0 --m repro --adversary-with-y --checkpoint std/std_resnet18_utkface_race_seed1_best.pth.tar --gpu 2 > log/utkface_race_border_eo.log 2>&1 &
# CIFAR-10S
nohup python3 train.py --dataset cifar10s --result-dir border --reprogram-size 32 --epochs 20 --adversary-with-logits --lmbda 10.0 --m repro --adversary-with-y --checkpoint std/std_resnet18_cifar10s_seed1_best.pth.tar --gpu 0 > log/cifar10s_border_eo.log 2>&1 &
```

Step 3: Reprogramming with Patch Trigger

```bash
# CelebA
nohup python3 train.py --dataset celeba --result-dir patch --reprogram-size 90 --epochs 20 --adversary-with-logits --lmbda 10.0 --m rpatch --adversary-with-y --checkpoint std/std_resnet18_celeba_seed1_best.pth.tar --gpu 0 > log/celeba_patch_eo.log 2>&1 &
# UTKFace
nohup python3 train.py --dataset utkface --domain-attrs Age --result-dir patch --reprogram-size 80 --epochs 20 --adversary-with-logits --lmbda 10.0 --m rpatch --adversary-with-y --checkpoint std/std_resnet18_utkface_age_seed1_best.pth.tar --gpu 1 > log/utkface_age_patch_eo.log  2>&1 &
nohup python3 train.py --dataset utkface --domain-attrs Race --result-dir patch --reprogram-size 80 --epochs 20 --adversary-with-logits --lmbda 10.0 --m rpatch --adversary-with-y --checkpoint std/std_resnet18_utkface_race_seed1_best.pth.tar --gpu 2 > log/utkface_race_patch_eo.log 2>&1 &
# CIFAR-10S
nohup python3 train.py --dataset cifar10s --result-dir patch --reprogram-size 2 --epochs 20 --adversary-with-logits --lmbda 10.0 --m rpatch --adversary-with-y --checkpoint std/std_resnet18_cifar10s_seed1_best.pth.tar --gpu 3 > log/cifar10s_patch_eo.log 2>&1 &
```

#### FAAP

Step 1: Train deployed models

```bash
# CelebA
nohup python train_deployed_model.py --dataset celeba --img-size 224 --pretrained --gpu 0 > log/celeba_deployed.log 2>&1 &
# UTKFace
nohup python train_deployed_model.py --dataset utkface --sensitive age --img-size 224 --pretrained --gpu 0 > log/utkface_age_deployed.log 2>&1 &
nohup python train_deployed_model.py --dataset utkface --sensitive race --img-size 224 --pretrained --gpu 1 > log/utkface_race_deployed.log 2>&1 &
# CIFAR-10S
nohup python train_deployed_model.py --dataset cifar10s --img-size 32 --pretrained --gpu 1 > log/cifar10s_deployed.log 2>&1 &
```

Step 2: Training of FAAP

```bash
# CelebA
nohup python faap.py --dataset celeba --lr 5e-4 --epochs 50 --batch-size 64 --img-size 224 --gpu 0 > log/celeba.log 2>&1 &
# UTKFace Age
nohup python faap.py --dataset utkface --sensitive age --lr 5e-4 --epochs 50 --batch-size 64 --img-size 224 --gpu 1 > log/utkface_age.log 2>&1 &
# UTKFace Race
nohup python faap.py --dataset utkface --sensitive race --lr 5e-4 --epochs 50 --batch-size 64 --img-size 224 --gpu 2 > log/utkface_race.log 2>&1 &
# CIFAR-10S
nohup python faap.py --dataset cifar10s --lr 5e-4 --epochs 50 --batch-size 64 --img-size 32 --gpu 3 > log/cifar10s.log 2>&1 &
```

Step 3: Test with adversarial examples

```bash
# CelebA
python test_adversarial_examples.py --dataset celeba --lr 5e-4 --epochs 50 --batch-size 64 --img-size 224 --gpu 0
# UTKFace Age
python test_adversarial_examples.py --dataset utkface --sensitive age --lr 5e-4 --epochs 50 --batch-size 64 --img-size 224 --gpu 1
# UTKFace Race
python test_adversarial_examples.py --dataset utkface --sensitive race --lr 5e-4 --epochs 50 --batch-size 64 --img-size 224 --gpu 2
# CIFAR-10S
python test_adversarial_examples.py --dataset cifar10s --lr 5e-4 --epochs 50 --batch-size 64 --img-size 32 --gpu 3
```
