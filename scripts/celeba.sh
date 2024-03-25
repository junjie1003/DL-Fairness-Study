# ================================================================== Pre-processing ==================================================================

## Undersampling (US)
nohup python BM/train_celeba/train_celeba_us.py --target Blond_Hair --epochs 170 --lr 1e-4 --batch-size 128 --gpu 0 > log/us_celeba.log 2>&1 &

## Oversampling (OS)
nohup python BM/train_celeba/train_celeba_os.py --target Blond_Hair --epochs 4 --lr 1e-4 --batch-size 128 --gpu 0 > log/os_celeba.log 2>&1 &

## Upweighting (UW)
nohup python BM/train_celeba/train_celeba_uw.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --gpu 0 > log/uw_celeba.log 2>&1 &

## Bias Mimicking (BM)
nohup python BM/train_celeba/train_celeba_bm.py --target Blond_Hair --epochs 10 --lr 1e-4 --lr_layer 1e-4 --batch-size 128 --mode none --gpu 0 > log/bm_celeba.log 2>&1 &

# ================================================================== In-processing ==================================================================

## Adversarial Training (Adv)
nohup python BM/train_celeba/train_celeba_adv.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --training_ratio 2 --alpha 1 --gpu 0 > log/adv_celeba.log 2>&1 &

## Domain Independent Training (DI)
nohup python BM/train_celeba/train_celeba_di.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --gpu 0 > log/di_celeba.log 2>&1 &

## Bias-Contrastive and Bias-Balanced Learning (BC+BB)
nohup python BM/train_celeba/train_celeba_bc.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --gpu 0 > log/bc_celeba.log 2>&1 &

## FLAC
nohup python FLAC/train_celeba.py --target Blond_Hair --epochs 10 --lr 1e-3 --batch-size 128 --alpha 30000 --gpu 0 > log/flac_celeba.log 2>&1 &

## MMD-based Fair Distillation (MFD)
nohup python MFD/main.py --method kd_mfd --dataset celeba --target Blond_Hair --epochs 50 --lr 1e-3 --batch-size 128 --labelwise --lambf 7 --lambh 0 --no-annealing --img-size 176 --teacher-path scratch-celeba-lr0.001-bs128-epochs50-seed1/best_model.pt --gpu 0 > log/mfd_student_celeba.log 2>&1 &

## Fair Deep Feature Reweighting (FDR)
nohup python FDR/celeba.py --epochs 1000 --lr 1e-3 --batch-size -1 --constraint EO --alpha 2 --gpu 0 > log/fdr_celeba.log 2>&1 &

# ================================================================== Post-processing ==================================================================

## FairReprogram (FR)
### Reprogramming with Border Trigger
nohup python FR/train.py --dataset celeba --rmethod repro --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 184 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_std-celeba-lr0.0001-bs64-epochs100-seed1 --gpu 5 > log/fr_border_celeba.log 2>&1 &
### Reprogramming with Patch Trigger
nohup python FR/train.py --dataset celeba --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 90 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_std-celeba-lr0.0001-bs64-epochs100-seed1 --gpu 7 > log/fr_patch_celeba.log 2>&1 &

## Fairness-Aware Adversarial Perturbation (FAAP)
nohup python FAAP/faap.py --dataset celeba --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-celeba-lr0.001-bs64-epochs100-seed1 --gpu 4 > log/faap_celeba.log 2>&1 &
