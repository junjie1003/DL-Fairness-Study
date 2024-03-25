# ================================================================== Pre-processing ==================================================================

## Undersampling (US)
python BM/train_cifar10/train_cifar10_us.py --epochs 2000 --lr 0.1 --batch-size 128 --checkpoint

## Oversampling (OS)
python BM/train_cifar10/train_cifar10_os.py --epochs 100 --lr 0.1 --batch-size 128 --checkpoint

## Upweighting (UW)
python BM/train_cifar10/train_cifar10_uw.py --epochs 200 --lr 0.01 --batch-size 128 --checkpoint

## Bias Mimicking (BM)
python BM/train_cifar10/train_cifar10_bm.py --epochs 200 --lr 0.1 --batch-size 128 --mode none --checkpoint

# ================================================================== In-processing ==================================================================

## Adversarial Training (Adv)
python BM/train_cifar10/train_cifar10_adv.py --epochs 200 --lr 0.01 --batch-size 128 --training_ratio 3 --alpha 1 --checkpoint

## Domain Independent Training (DI)
python BM/train_cifar10/train_cifar10_di.py --epochs 200 --lr 0.1 --batch-size 128 --checkpoint

## Bias-Contrastive and Bias-Balanced Learning (BC+BB)
python BM/train_cifar10/train_cifar10_bc.py --epochs 200 --lr 0.1 --batch-size 128 --checkpoint

## FLAC
python FLAC/train_cifar10s.py --epochs 200 --lr 1e-3 --batch-size 128 --alpha 1000 --checkpoint

## MMD-based Fair Distillation (MFD)
python MFD/main.py --method kd_mfd --dataset cifar10s --epochs 100 --lr 1e-3 --batch-size 128 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 32 --model-path kd_mfd-cifar10s-lr0.001-bs128-epochs100-seed1-labelwise-temp3-lambh0.0-lambf3.0-fixedlamb-rbf-sigma1.0/best_model.pt --checkpoint

## Fair Deep Feature Reweighting (FDR)
python FDR/cifar10s.py --epochs 1000 --lr 1e-3 --batch-size -1 --constraint EO --alpha 2 --checkpoint

# ================================================================== Post-processing ==================================================================

## FairReprogram (FR)
### Reprogramming with Border Trigger
python FR/train.py --dataset cifar10s --rmethod repro --epochs 20 --lr 1e-4 --batch-size 512 --reprogram-size 32 --adversary-with-y --adversary-with-logits --adversary-lr 0.001 --lmbda 10.0 --model-path fr_border-cifar10s-lr0.0001-bs512-epochs20-seed1-lambda10.0-eo-size32 --checkpoint
### Reprogramming with Patch Trigger
python FR/train.py --dataset cifar10s --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 512 --reprogram-size 2 --adversary-with-y --adversary-with-logits --adversary-lr 0.001 --lmbda 10.0 --model-path fr_patch-cifar10s-lr0.0001-bs512-epochs20-seed1-lambda10.0-eo-size2 --checkpoint

## Fairness-Aware Adversarial Perturbation (FAAP)
python FAAP/test_adversarial_examples.py --dataset cifar10s --epochs 50 --lr 5e-4 --batch-size 64 --img-size 32 --model-path deployed-cifar10s-lr0.001-bs64-epochs100-seed1
