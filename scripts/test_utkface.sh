# ================================================================== Pre-processing ==================================================================

## Undersampling (US)
python BM/train_utk_face/train_utk_face_us.py --sensitive age --epochs 400 --epochs_extra 400 --lr 1e-5 --batch-size 128 --checkpoint
python BM/train_utk_face/train_utk_face_us.py --sensitive race --epochs 120 --epochs_extra 120 --lr 1e-4 --batch-size 128 --checkpoint

## Oversampling (OS)
python BM/train_utk_face/train_utk_face_os.py --sensitive age --epochs 7 --lr 1e-4 --batch-size 128 --checkpoint
python BM/train_utk_face/train_utk_face_os.py --sensitive race --epochs 10 --lr 1e-4 --batch-size 128 --checkpoint

## Upweighting (UW)
python BM/train_utk_face/train_utk_face_uw.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --checkpoint
python BM/train_utk_face/train_utk_face_uw.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --checkpoint

## Bias Mimicking (BM)
python BM/train_utk_face/train_utk_face_bm.py --sensitive age --epochs 20 --lr 5e-4 --batch-size 128 --mode none --checkpoint
python BM/train_utk_face/train_utk_face_bm.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --mode none --checkpoint

# ================================================================== In-processing ==================================================================

## Adversarial Training (Adv)
python BM/train_utk_face/train_utk_face_adv.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --training_ratio 3 --alpha 1 --checkpoint
python BM/train_utk_face/train_utk_face_adv.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --training_ratio 3 --alpha 1 --checkpoint

## Domain Independent Training (DI)
python BM/train_utk_face/train_utk_face_di.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --checkpoint
python BM/train_utk_face/train_utk_face_di.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --checkpoint

## Bias-Contrastive and Bias-Balanced Learning (BC+BB)
python BM/train_utk_face/train_utk_face_bc.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --checkpoint
python BM/train_utk_face/train_utk_face_bc.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --checkpoint

## FLAC
python FLAC/train_utk_face.py --sensitive age --epochs 20 --lr 1e-3 --batch-size 128 --alpha 1000 --checkpoint
python FLAC/train_utk_face.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --alpha 1000 --checkpoint

## MMD-based Fair Distillation (MFD)
python MFD/main.py --method kd_mfd --dataset utkface --sensitive age --epochs 100 --lr 1e-3 --batch-size 128 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --model-path kd_mfd-utkface_age-lr0.001-bs128-epochs100-seed1-labelwise-temp3-lambh0.0-lambf3.0-fixedlamb-rbf-sigma1.0/best_model.pt --checkpoint
python MFD/main.py --method kd_mfd --dataset utkface --sensitive race --epochs 100 --lr 1e-3 --batch-size 128 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --model-path kd_mfd-utkface_race-lr0.001-bs128-epochs100-seed1-labelwise-temp3-lambh0.0-lambf3.0-fixedlamb-rbf-sigma1.0/best_model.pt --checkpoint

## Fair Deep Feature Reweighting (FDR)
python FDR/utkface.py --sensitive age --epochs 1500 --lr 5e-3 --batch-size -1 --constraint EO --alpha 2 --gpu 7 --checkpoint
python FDR/utkface.py --sensitive race --epochs 1500 --lr 1e-3 --batch-size -1 --constraint EO --alpha 2 --gpu 7 --checkpoint

# ================================================================== Post-processing ==================================================================

## FairReprogram (FR)
### Reprogramming with Border Trigger
python FR/train.py --dataset utkface --sensitive age --rmethod repro --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 184 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_border-utkface_age-lr0.0001-bs128-epochs20-seed1-lambda10.0-eo-size184 --checkpoint
python FR/train.py --dataset utkface --sensitive race --rmethod repro --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 184 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_border-utkface_race-lr0.0001-bs128-epochs20-seed1-lambda10.0-eo-size184 --checkpoint

### Reprogramming with Patch Trigger
python FR/train.py --dataset utkface --sensitive age --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 512 --reprogram-size 60 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 50.0 --model-path fr_patch-utkface_age-lr0.0001-bs512-epochs20-seed1-lambda50.0-eo-size60 --checkpoint
python FR/train.py --dataset utkface --sensitive race --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 80 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_patch-utkface_race-lr0.0001-bs128-epochs20-seed1-lambda10.0-eo-size80 --checkpoint

## Fairness-Aware Adversarial Perturbation (FAAP)
python FAAP/test_adversarial_examples.py --dataset utkface --sensitive age --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-utkface_age-lr0.001-bs64-epochs100-seed1 --checkpoint
python FAAP/test_adversarial_examples.py --dataset utkface --sensitive race --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-utkface_race-lr0.001-bs64-epochs100-seed1 --checkpoint
