import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Fairness")

    parser.add_argument("--gpu", default=0, type=int, help="CUDA visible device")
    parser.add_argument("--date", default="230929", type=str, help="experiment date")
    parser.add_argument("--model", default="resnet18", type=str, choices=["resnet18"])
    parser.add_argument("--data-dir", default="../data", type=str, help="data directory")
    parser.add_argument("--target", default="Blond_Hair", type=str, help="target attribute for celeba")
    parser.add_argument("--sensitive", default="race", type=str, help="sensitive attribute for utkface")
    parser.add_argument("--pretrained", default=False, action="store_true", help="load pre-trained model")
    parser.add_argument("--dataset", default="celeba", type=str, choices=["celeba", "utkface", "cifar10s"])
    parser.add_argument("--save-dir", default="./saved_models", type=str, help="directory to save trained models)")

    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--seed", default=1, type=int, help="seed for randomness")
    parser.add_argument("--batch-size", default=64, type=int, help="mini batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--img-size", default=224, type=int, help="image size for preprocessing")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args
