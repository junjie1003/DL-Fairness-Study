# ================================================================== Pre-processing ==================================================================

## Undersampling (US)
nohup python BM/train_cifar10/train_cifar10_us.py --epochs 2000 --lr 0.1 --batch-size 128 --gpu 1 > log/us_cifar10s.log 2>&1 &

## Oversampling (OS)
nohup python BM/train_cifar10/train_cifar10_os.py --epochs 100 --lr 0.1 --batch-size 128 --gpu 4 > log/os_cifar10s.log 2>&1 &

## Upweighting (UW)
nohup python BM/train_cifar10/train_cifar10_uw.py --epochs 200 --lr 0.01 --batch-size 128 --gpu 4 > log/uw_cifar10s.log 2>&1 &

## Bias Mimicking (BM)
nohup python BM/train_cifar10/train_cifar10_bm.py --epochs 200 --lr 0.1 --batch-size 128 --mode none --gpu 5 > log/bm_cifar10s.log 2>&1 &

# ================================================================== In-processing ==================================================================

## Adversarial Training (Adv)
nohup python BM/train_cifar10/train_cifar10_adv.py --epochs 200 --lr 0.01 --batch-size 128 --training_ratio 3 --alpha 1 --gpu 7 > log/adv_cifar10s.log 2>&1 &

## Domain Independent Training (DI)
nohup python BM/train_cifar10/train_cifar10_di.py --epochs 200 --lr 0.1 --batch-size 128 --gpu 7 > log/di_cifar10s.log 2>&1 &

## Bias-Contrastive and Bias-Balanced Learning (BC+BB)
nohup python BM/train_cifar10/train_cifar10_bc.py --epochs 200 --lr 0.1 --batch-size 128 --gpu 5 > log/bc_cifar10s.log 2>&1 &

## FLAC
nohup python FLAC/train_cifar10s.py --epochs 200 --lr 1e-3 --batch-size 128 --alpha 1000 --gpu 6 > log/flac_cifar10s.log 2>&1 &

## MMD-based Fair Distillation (MFD)
nohup python MFD/main.py --method kd_mfd --dataset cifar10s --epochs 100 --lr 1e-3 --batch-size 128 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 32 --teacher-path scratch-cifar10s-lr0.001-bs128-epochs100-seed1-pretrained/base_model.pt --gpu 7 > log/mfd_student_cifar10s.log 2>&1 &

## Fair Deep Feature Reweighting (FDR)
nohup python FDR/cifar10s.py --epochs 1000 --lr 1e-3 --batch-size -1 --constraint EO --alpha 2 --gpu 2 > log/fdr_cifar10s.log 2>&1 &

# ================================================================== Post-processing ==================================================================

## FairReprogram (FR)
### Reprogramming with Border Trigger
nohup python FR/train.py --dataset cifar10s --rmethod repro --epochs 20 --lr 1e-4 --batch-size 512 --reprogram-size 32 --adversary-with-y --adversary-with-logits --adversary-lr 0.001 --lmbda 10.0 --model-path fr_std-cifar10s-lr0.0001-bs64-epochs100-seed1 --gpu 2 > log/fr_border_cifar10s.log 2>&1 &
### Reprogramming with Patch Trigger
nohup python FR/train.py --dataset cifar10s --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 512 --reprogram-size 2 --adversary-with-y --adversary-with-logits --adversary-lr 0.001 --lmbda 10.0 --model-path fr_std-cifar10s-lr0.0001-bs64-epochs100-seed1 --gpu 3 > log/fr_patch_cifar10s.log 2>&1 &

## Fairness-Aware Adversarial Perturbation (FAAP)
nohup python FAAP/faap.py --dataset cifar10s --epochs 50 --lr 5e-4 --batch-size 64 --img-size 32 --model-path deployed-cifar10s-lr0.001-bs64-epochs100-seed1 --gpu 1 > log/faap_cifar10s.log 2>&1 &
