import os
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from arguments import get_args
from models import Discriminator, Generator, ResNet18

sys.path.insert(1, "/root/study")
from datasets.celeba import get_celeba
from datasets.utkface import get_utkface
from datasets.cifar10s import get_cifar10s
from helper import set_seed, make_log_name

args = get_args()


# Custom weights initialization called on netG and netD
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class FAAP:
    def __init__(self, model, device, save_dir, log_name, epochs, channels=3, box_min=-1, box_max=1):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_name = log_name
        self.epochs = epochs
        self.box_min = box_min
        self.box_max = box_max

        self.netG = Generator(channels).to(device)
        self.netD = Discriminator().to(device)

        # Initialize all weights
        self.netG.apply(init_weights)
        self.netD.apply(init_weights)

        # Initialize all optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=args.lr)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.2 * args.lr, weight_decay=1e-4)

    def train_batch(self, images, labels, biases):
        n = labels.shape[0]

        # Optimize discriminator D
        perturbation = self.netG(images)

        # Add a clipping trick
        perturbation_min = -0.05
        perturbation_max = 0.05
        perturbed_images = torch.clamp(perturbation, perturbation_min, perturbation_max) + images
        perturbed_images = torch.clamp(perturbed_images, self.box_min, self.box_max)

        with torch.no_grad():
            features = self.model.feature_extractor(perturbed_images)

        # Calculate discriminator loss
        self.optimizer_D.zero_grad()
        logits_bias = self.netD(features.detach())
        loss_D = F.cross_entropy(logits_bias, biases)
        loss_D.backward()
        self.optimizer_D.step()

        # Optimize generator G
        self.optimizer_G.zero_grad()

        # Calculate fairness loss
        logits_bias = self.netD(features)
        probs_bias = F.softmax(logits_bias, dim=1)
        # Calculate the entropy by PyTorch
        entropy = Categorical(probs=probs_bias).entropy()
        entropy_avg = torch.sum(entropy) / n
        loss_G_fair = -0.4 * F.cross_entropy(logits_bias, biases) - 0.1 * entropy_avg
        # print("loss_G_fair:", loss_G_fair)

        # Calculate target label prediction loss
        with torch.no_grad():
            logits_label = self.model.label_predictor(features.view(features.size(0), -1))

        loss_G_target = F.cross_entropy(logits_label, labels)
        # print("loss_G_target:", loss_G_target)

        # Get total loss of generator G
        loss_G = loss_G_fair + 0.7 * loss_G_target
        loss_G.backward()
        self.optimizer_G.step()

        return loss_D.item(), loss_G.item()

    def train(self, train_dataloader):
        for epoch in range(1, self.epochs + 1):
            loss_D_sum = 0
            loss_G_sum = 0

            for images, labels, biases in train_dataloader:
                # if epoch == 1:
                #     print(images, images.shape)
                images, labels, biases = images.to(self.device), labels.to(self.device), biases.to(self.device)

                loss_D_batch, loss_G_batch = self.train_batch(images, labels, biases)

                loss_D_sum += loss_D_batch
                loss_G_sum += loss_G_batch

            num_batch = len(train_dataloader)
            print("Epoch %d:\nloss_D: %.3f, loss_G: %.3f\n" % (epoch, loss_D_sum / num_batch, loss_G_sum / num_batch))

            # Save generators
            if epoch > self.epochs - 10:
                netG_filename = os.path.join(self.save_dir, f"{self.log_name}_epoch{str(epoch)}.pt")
                torch.save(self.netG.state_dict(), netG_filename)


if __name__ == "__main__":
    print(args)
    set_seed(args.seed)

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3, sci_mode=False)

    save_dir = Path(f"{args.save_dir}/{args.date}")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_name_d = make_log_name(args, model_name="deployed_model")
    log_name_g = make_log_name(args, model_name="generator")
    num_classes = 10 if args.dataset == "cifar10s" else 2

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    data_dir = f"{args.data_dir}/{args.dataset}"
    if args.dataset == "celeba":
        train_dataset, train_loader = get_celeba(
            root=args.data_dir, split="train", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_celeba(
            root=args.data_dir, split="valid", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_celeba(
            root=args.data_dir, split="test", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "utkface":
        train_dataset, train_loader = get_utkface(
            root=data_dir, split="train", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_utkface(
            root=data_dir, split="valid", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_utkface(
            root=data_dir, split="test", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "cifar10s":
        train_dataset, train_loader = get_cifar10s(root=data_dir, split="train", img_size=args.img_size, batch_size=args.batch_size)
        valid_dataset, valid_loader = get_cifar10s(root=data_dir, split="valid", img_size=args.img_size, batch_size=args.batch_size)
        test_dataset, test_loader = get_cifar10s(root=data_dir, split="test", img_size=args.img_size, batch_size=args.batch_size)

    ckpt_path = os.path.join(save_dir, log_name_d + ".pt")

    deployed_model = ResNet18(num_classes=num_classes, pretrained=False).to(device)
    deployed_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    deployed_model.eval()

    start_time = time.time()

    if args.dataset == "celeba":
        faap = FAAP(deployed_model, device, save_dir, log_name_g, args.epochs, channels=3, box_min=0, box_max=1)
    elif args.dataset == "utkface":
        faap = FAAP(deployed_model, device, save_dir, log_name_g, args.epochs, channels=3, box_min=0, box_max=1)
    elif args.dataset == "cifar10s":
        faap = FAAP(deployed_model, device, save_dir, log_name_g, args.epochs, channels=3, box_min=-1, box_max=1)

    faap.train(train_loader)

    time_per_epoch = (time.time() - start_time) / args.epochs
    print("Time per epoch: %.1f" % time_per_epoch)
