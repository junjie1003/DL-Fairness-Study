echo "A Quick Start on UTKFace Race Dataset"
echo ""
echo "Pre-processing Methods"
echo ""
echo "Bias Mimicking (BM)"
python BM/train_utk_face/train_utk_face_bm.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --mode none --gpu 0 && python BM/train_utk_face/train_utk_face_bm.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --mode none --checkpoint
echo ""
echo "In-processing Methods"
echo ""
echo "FLAC"
python FLAC/train_utk_face.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --alpha 1000 --gpu 1 && python FLAC/train_utk_face.py --sensitive race --epochs 20 --lr 1e-3 --batch-size 128 --alpha 1000 --checkpoint
echo ""
echo "Post-processing Methods"
echo ""
echo "FAAP"
python FAAP/faap.py --dataset utkface --sensitive race --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-utkface_race-lr0.001-bs64-epochs100-seed1 --gpu 2 && python FAAP/test_adversarial_examples.py --dataset utkface --sensitive race --epochs 50 --lr 5e-4 --batch-size 64 --img-size 224 --model-path deployed-utkface_race-lr0.001-bs64-epochs100-seed1 --checkpoint
