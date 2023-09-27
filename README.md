# An Empirical Study of Image Fairness

## Fairness Improvement Methods

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
cd FLAC

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
cd MFD

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
cd FDR

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
