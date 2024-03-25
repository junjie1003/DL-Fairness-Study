# ================================================================== Pre-processing ==================================================================

## Undersampling (US)
nohup python BM/train_utk_face/train_utk_face_us.py --sensitive age --epochs 400 --epochs_extra 400 --lr 1e-5 --batch-size 128 --gpu 0 > log/us_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_us.py --sensitive race --epochs 120 --epochs_extra 120 --lr 1e-4 --batch-size 128 --gpu 1 > log/us_utkface_race.log 2>&1 &

## Oversampling (OS)
nohup python BM/train_utk_face/train_utk_face_os.py --sensitive age --epochs 7 --lr 1e-4 --batch-size 128 --gpu 0 > log/os_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_os.py --sensitive race --epochs 10 --lr 1e-4 --batch-size 128 --gpu 1 > log/os_utkface_race.log 2>&1 &

## Upweighting (UW)
nohup python BM/train_utk_face/train_utk_face_uw.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --gpu 0 > log/uw_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_uw.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --gpu 1 > log/uw_utkface_race.log 2>&1 &

## Bias Mimicking (BM)
nohup python BM/train_utk_face/train_utk_face_bm.py --sensitive age --epochs 20 --lr 5e-4 --batch-size 128 --mode none --gpu 0 > log/bm_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_bm.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --mode none --gpu 2 > log/bm_utkface_race.log 2>&1 &

# ================================================================== In-processing ==================================================================

## Adversarial Training (Adv)
nohup python BM/train_utk_face/train_utk_face_adv.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --training_ratio 3 --alpha 1 --gpu 0 > log/adv_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_adv.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --training_ratio 3 --alpha 1 --gpu 2 > log/adv_utkface_race.log 2>&1 &

## Domain Independent Training (DI)
nohup python BM/train_utk_face/train_utk_face_di.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --gpu 0 > log/di_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_di.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --gpu 2 > log/di_utkface_race.log 2>&1 &

## Bias-Contrastive and Bias-Balanced Learning (BC+BB)
nohup python BM/train_utk_face/train_utk_face_bc.py --sensitive age --epochs 20 --lr 1e-4 --batch-size 128 --gpu 0 > log/bc_utkface_age.log 2>&1 &
nohup python BM/train_utk_face/train_utk_face_bc.py --sensitive race --epochs 20 --lr 1e-4 --batch-size 128 --gpu 2 > log/bc_utkface_race.log 2>&1 &

## FLAC
nohup python FLAC/train_utk_face.py --sensitive age --epochs 20 --lr 1e-3 --batch-size 128 --alpha 1000 --gpu 0 > log/flac_utkface_age.log 2>&1 &
nohup python FLAC/train_utk_face.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --alpha 1000 --gpu 2 > log/flac_utkface_race.log 2>&1 &

## MMD-based Fair Distillation (MFD)
nohup python MFD/main.py --method kd_mfd --dataset utkface --sensitive age --epochs 100 --lr 1e-3 --batch-size 128 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --teacher-path scratch-utkface_age-lr0.001-bs128-epochs100-seed1-pretrained/best_model.pt --gpu 1 > log/mfd_student_utkface_age.log 2>&1 &
nohup python MFD/main.py --method kd_mfd --dataset utkface --sensitive race --epochs 100 --lr 1e-3 --batch-size 128 --labelwise --lambf 3 --lambh 0 --no-annealing --img-size 176 --teacher-path scratch-utkface_race-lr0.001-bs128-epochs100-seed1-pretrained/best_model.pt --gpu 2 > log/mfd_student_utkface_race.log 2>&1 &

## Fair Deep Feature Reweighting (FDR)
nohup python FDR/utkface.py --sensitive age --epochs 1500 --lr 5e-3 --batch-size -1 --constraint EO --alpha 2 --gpu 0 > log/fdr_utkface_age.log 2>&1 &
nohup python FDR/utkface.py --sensitive race --epochs 1500 --lr 1e-3 --batch-size -1 --constraint EO --alpha 2 --gpu 2 > log/fdr_utkface_race.log 2>&1 &

# ================================================================== Post-processing ==================================================================

## FairReprogram (FR)
### Reprogramming with Border Trigger
nohup python FR/train.py --dataset utkface --sensitive age --rmethod repro --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 184 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_std-utkface_age-lr0.0001-bs64-epochs100-seed1 --gpu 4 > log/fr_border_utkface_age.log 2>&1 &
nohup python FR/train.py --dataset utkface --sensitive race --rmethod repro --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 184 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_std-utkface_race-lr0.0001-bs64-epochs100-seed1 --gpu 4 > log/fr_border_utkface_race.log 2>&1 &

### Reprogramming with Patch Trigger
nohup python FR/train.py --dataset utkface --sensitive age --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 512 --reprogram-size 60 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 50.0 --model-path fr_std-utkface_age-lr0.0001-bs64-epochs100-seed1 --gpu 5 > log/fr_patch_utkface_age.log 2>&1 &
nohup python FR/train.py --dataset utkface --sensitive race --rmethod rpatch --epochs 20 --lr 1e-4 --batch-size 128 --reprogram-size 80 --adversary-with-y --adversary-with-logits --adversary-lr 0.01 --lmbda 10.0 --model-path fr_std-utkface_race-lr0.0001-bs64-epochs100-seed1 --gpu 5 > log/fr_patch_utkface_race.log 2>&1 &

## Fairness-Aware Adversarial Perturbation (FAAP)
nohup python FAAP/faap.py --dataset utkface --sensitive age --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-utkface_age-lr0.001-bs64-epochs100-seed1 --gpu 5 > log/faap_utkface_age.log 2>&1 &
nohup python FAAP/faap.py --dataset utkface --sensitive race --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-utkface_race-lr0.001-bs64-epochs100-seed1 --gpu 4 > log/faap_utkface_race.log 2>&1 &
