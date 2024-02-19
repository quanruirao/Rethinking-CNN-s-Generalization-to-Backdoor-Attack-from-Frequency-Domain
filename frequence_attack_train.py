import copy
import json
import os
import random
import shutil
from time import time

import config
from frequence_change import DCT
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18, VGG, MobileNetV2
from networks.models import Denormalizer, Normalizer
from tensorboardX import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from fractions import Fraction

def get_model(opt):
    netC = None

    if opt.dataset == "cifar10":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = ResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = VGG("VGG16", opt.num_classes).to(opt.device)
        # netC = MobileNetV2(opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18(num_classes=opt.num_classes,data_name = "celeba").to(opt.device)
        # netC = VGG("VGG16", opt.num_classes, is_celeba=True).to(opt.device)
        # netC = MobileNetV2(opt.num_classes, is_celeba=True).to(opt.device)
    if opt.dataset == "mnist":
        netC = ResNet18(num_classes=opt.num_classes, in_channel=1).to(opt.device)
        # netC = VGG("VGG16", opt.num_classes, in_channel=1).to(opt.device)
        # netC = MobileNetV2(opt.num_classes, in_channel=1).to(opt.device)


    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    # optimizerC = torch.optim.Adam(netC.parameters(), opt.lr_C, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt, tearget_train_datasets):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0


    identity_grid = torch.zeros((1, 32, 32), dtype=torch.uint8)
    identity_grid[0, -3:, -3:] = 255

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        if opt.dataset == "mnist":
            is_mnist = True
        else:
            is_mnist = False
        rd = random.randint(0,len(tearget_train_datasets)-1)

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        inputs_bd = DCT(copy.deepcopy(tearget_train_datasets[rd]), copy.deepcopy(inputs[:num_bd]),
                        img_size=opt.input_height,dct_size=int(inputs.shape[2] * opt.dctsize),is_mnist=is_mnist,intensity=opt.intensity)

        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label

        total_inputs = torch.cat([inputs_bd,  inputs[(num_bd ) :]], dim=0)
        total_inputs = transforms(total_inputs)

        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time()

        total_preds = netC(total_inputs)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd
        total_bd += num_bd
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd ) :], dim=1) == total_targets[(num_bd ) :]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)


        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample

        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
        )

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd}, epoch
        )

    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    tf_writer,
    epoch,
    opt,
    tearget_test_datasets):

    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            preds_clean = netC(inputs)

            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            rd = random.randint(0, len(tearget_test_datasets)-1)

            if opt.dataset == "mnist":
                is_mnist = True
            else:
                is_mnist = False

            inputs_bd = DCT(copy.deepcopy(tearget_test_datasets[rd]), copy.deepcopy(inputs),
                            img_size=opt.input_height,dct_size=int(inputs.shape[2] * opt.dctsize),is_mnist=is_mnist,intensity=opt.intensity)

            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label

            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample


            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd, best_bd_acc
            )

            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc) or acc_bd > best_bd_acc:
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch_current": epoch
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item()
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10

    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    opt.dctsize = float(Fraction(opt.dctsize))

    # Dataset
    train_dl,transforms = get_dataloader(opt, True)
    test_dl,transforms = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            epoch_current = state_dict["epoch_current"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    target_train_images = []

    for target_train_batch_idx, (target_train_data, target_train_targets) in enumerate(train_dl):

        target_train_indices = torch.nonzero(target_train_targets == opt.target_label).squeeze()

        if target_train_indices.numel() > 0:
            target_train_batch_images = target_train_data[target_train_indices]
            target_train_images.append(target_train_batch_images)

    target_train_images = torch.cat(target_train_images, dim=0)
    tearget_train_datasets = target_train_images.data

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt,tearget_train_datasets)
        best_clean_acc, best_bd_acc= eval(
            netC=netC,
            optimizerC=optimizerC,
            schedulerC=schedulerC,
            test_dl=test_dl,
            best_clean_acc=best_clean_acc,
            best_bd_acc=best_bd_acc,
            tf_writer=tf_writer,
            epoch=epoch,
            opt=opt,
            tearget_test_datasets= tearget_train_datasets,
        )


if __name__ == "__main__":
    main()
