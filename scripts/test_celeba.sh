# ================================================================== Pre-processing ==================================================================

## Undersampling (US)
python BM/train_celeba/train_celeba_us.py --target Blond_Hair --epochs 170 --lr 1e-4 --batch-size 128 --checkpoint

## Oversampling (OS)
python BM/train_celeba/train_celeba_os.py --target Blond_Hair --epochs 4 --lr 1e-4 --batch-size 128 --checkpoint

## Upweighting (UW)
python BM/train_celeba/train_celeba_uw.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --checkpoint

## Bias Mimicking (BM)
python BM/train_celeba/train_celeba_bm.py --target Blond_Hair --epochs 10 --lr 1e-4 --lr_layer 1e-4 --batch-size 128 --mode none --checkpoint

# ================================================================== In-processing ==================================================================

## Adversarial Training (Adv)
python BM/train_celeba/train_celeba_adv.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --training_ratio 2 --alpha 1 --checkpoint

## Domain Independent Training (DI)
python BM/train_celeba/train_celeba_di.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --checkpoint

## Bias-Contrastive and Bias-Balanced Learning (BC+BB)
python BM/train_celeba/train_celeba_bc.py --target Blond_Hair --epochs 10 --lr 1e-4 --batch-size 128 --checkpoint

## FLAC
python FLAC/train_celeba.py --target Blond_Hair --epochs 10 --lr 1e-3 --batch-size 128 --alpha 30000 --checkpoint

## MMD-based Fair Distillation (MFD)
python MFD/main.py --method kd_mfd --dataset celeba --target Blond_Hair --epochs 50 --lr 1e-3 --batch-size 128 --labelwise --lambf 7 --lambh 0 --no-annealing --img-size 176 --model-path kd_mfd-celeba-lr0.001-bs128-epochs50-seed1-labelwise-temp3-lambh0.0-lambf7.0-fixedlamb-rbf-sigma1.0/best_model.pt --checkpoint

## Fair Deep Feature Reweighting (FDR)
python FDR/celeba.py --epochs 1000 --lr 1e-3 --batch-size -1 --constraint EO --alpha 2 --checkpoint

# ================================================================== Post-processing ==================================================================

## FairReprogram (FR)
### Reprogramming with Border Trigger
python FR/train.py --dataset celeba --rmethod repro --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 184 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_border-celeba-lr0.0001-bs128-epochs20-seed1-lambda10.0-eo-size184 --checkpoint
### Reprogramming with Patch Trigger
python FR/train.py --dataset celeba --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 90 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_patch-celeba-lr0.0001-bs128-epochs20-seed1-lambda10.0-eo-size90 --checkpoint

## Fairness-Aware Adversarial Perturbation (FAAP)
python FAAP/test_adversarial_examples.py --dataset celeba --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-celeba-lr0.001-bs64-epochs100-seed1 
