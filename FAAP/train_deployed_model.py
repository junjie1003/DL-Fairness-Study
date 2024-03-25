import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

project_dir = "/root/DL-Fairness-Study"
sys.path.insert(1, os.path.join(project_dir, "FAAP"))

from models import ResNet18
from datasets.celeba import get_celeba
from datasets.utkface import get_utkface
from datasets.cifar10s import get_cifar10s

sys.path.insert(1, project_dir)
from helper import set_seed
from arguments import get_args

args = get_args()


def train(model, train_loader, criterion, optimizer, epoch, epochs):
    model.train()

    n = 0
    train_acc = 0.0
    train_loss = 0.0

    for images, labels, biases in train_loader:
        bsz = labels.shape[0]
        images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()

        optimizer.zero_grad()

        logits, _ = model(images)
        _, preds = torch.max(logits.data, 1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_acc += torch.sum(preds == labels).item()
        train_loss += loss.item() * bsz
        n += bsz

    train_acc /= n
    train_loss /= n

    print("Training Epoch: %d/%d\tLoss: %.3f\tAccuracy: %.3f" % (epoch, epochs, train_loss, train_acc))

    return train_loss, train_acc


def test(model, test_loader):
    model.eval()

    n = 0
    test_acc = 0.0

    with torch.no_grad():
        for images, labels, biases in test_loader:
            bsz = labels.shape[0]
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()

            logits, _ = model(images)
            _, preds = torch.max(logits.data, 1)

            test_acc += torch.sum(preds == labels).item()
            n += bsz

    test_acc /= n

    return test_acc


def main():
    print(args)
    set_seed(args.seed)

    if args.dataset in ["celeba", "cifar10s"]:
        exp_name = f"deployed-{args.dataset}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"
    elif args.dataset == "utkface":
        exp_name = f"deployed-utkface_{args.sensitive}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-seed{args.seed}"
    save_dir = Path(f"{project_dir}/checkpoints/{exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = f"{project_dir}/data"
    num_classes = 10 if args.dataset == "cifar10s" else 2

    # Load datasets
    dataset_dir = f"{data_dir}/{args.dataset}"
    if args.dataset == "celeba":
        train_dataset, train_loader = get_celeba(
            root=data_dir, split="train", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_celeba(
            root=data_dir, split="valid", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_celeba(
            root=data_dir, split="test", target_attr=args.target, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "utkface":
        train_dataset, train_loader = get_utkface(
            root=dataset_dir, split="train", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        valid_dataset, valid_loader = get_utkface(
            root=dataset_dir, split="valid", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
        test_dataset, test_loader = get_utkface(
            root=dataset_dir, split="test", bias_attr=args.sensitive, img_size=args.img_size, batch_size=args.batch_size
        )
    elif args.dataset == "cifar10s":
        train_dataset, train_loader = get_cifar10s(root=dataset_dir, split="train", img_size=args.img_size, batch_size=args.batch_size)
        valid_dataset, valid_loader = get_cifar10s(root=dataset_dir, split="valid", img_size=args.img_size, batch_size=args.batch_size)
        test_dataset, test_loader = get_cifar10s(root=dataset_dir, split="test", img_size=args.img_size, batch_size=args.batch_size)

    # Train the deployed model
    deployed_model = ResNet18(num_classes=num_classes, pretrained=args.pretrained).cuda()
    criterion = nn.CrossEntropyLoss()

    decay_epochs = [args.epochs // 4, args.epochs // 2, args.epochs * 3 // 4]
    print(decay_epochs)
    optimizer = torch.optim.Adam(deployed_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    accs = []
    patience = 0
    best_acc = 0
    best_epoch = 0
    best_deployed_model = None
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")

        train(deployed_model, train_loader, criterion, optimizer, epoch, args.epochs)
        scheduler.step()
        valid_acc = test(deployed_model, valid_loader)
        accs.append(valid_acc)

        print("Valid Epoch: %d/%d\tAccuracy: %.3f" % (epoch, args.epochs, valid_acc))

        if valid_acc > best_acc and epoch > 5:
            best_epoch = epoch
            best_acc = valid_acc
            best_deployed_model = deepcopy(deployed_model)

        # Early Stop
        if (epoch > 20) and (accs[-1] <= accs[-2]):
            patience += 1
            if patience > 5:
                break
        else:
            patience = 0

    print("Best Epoch: %d/%d\tBest Accuracy: %.3f" % (best_epoch, args.epochs, best_acc))

    # Test the best deployed model
    test_acc = test(best_deployed_model, test_loader)
    print("Test Accuracy: %.3f" % (test_acc))

    # Save the best model
    save_file = os.path.join(f"{save_dir}/best_model.pt")
    torch.save(best_deployed_model.state_dict(), save_file)


if __name__ == "__main__":
    main()
