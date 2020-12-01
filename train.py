import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from progress.bar import Bar as Bar
from torch.utils.data import DataLoader

from losses import JointsMSELoss
from model.hourglass import HourglassNet
from utils.data_preprocessing import EyeLandmarksDataset, TrainDataset, TestDataset
from utils.metrics import heatmap_metrics_eval, AverageMeter
from utils.optimizer import get_optimizer
from utils.tools import adjust_learning_rate, save_checkpoint, save_checkpoint_during_time, to_numpy


def train(train_loader, model, criterion, optimizer,
          train_size=1000, print_step=10, save_step=1200, best_acc=None,
          checkpoint_path=r""):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Train', max=train_size)
    iter_train_loader = iter(train_loader)
    for i in range(train_size):
        (images, heat_maps, meta) = next(iter_train_loader)
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        images = images.to(device)
        heat_maps = heat_maps.to(device, non_blocking=True)

        # compute output
        output = model(images)

        loss = 0
        for o in output:
            loss += criterion(o, heat_maps)
        output = output[-1]

        loss.backward()
        optimizer.step()

        acc = heatmap_metrics_eval(to_numpy(output), to_numpy(heat_maps), to_numpy(meta["iris_diameter"]) / 2.0)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        acces.update(acc[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_step == 0:
            print(
                '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.2e} | Acc: {acc1: .3f}, {acc2: .3f}'.format(
                    batch=i + 1,
                    size=train_size,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc1=acc[0],
                    acc2=acc[1]
                ))
        bar.next()

        if (i + 1) % save_step == 0:
            print("Save checkpoint {}".format(checkpoint_path))
            save_checkpoint_during_time({
                'epoch': -1,
                'state_dict': model.state_dict(),
                'best_acc': acc[0],
                'optimizer': optimizer.state_dict(),
            }, checkpoint=checkpoint_path)

    bar.finish()
    return losses.avg, acces.avg, best_acc


def validate(val_loader, model, criterion,
             print_step=10, best_acc=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (images, heat_maps, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.to(device)
            heat_maps = heat_maps.to(device, non_blocking=True)

            # compute output
            output = model(images)

            loss = 0
            for o in output:
                loss += criterion(o, heat_maps)
            output = output[-1]

            acc = heatmap_metrics_eval(to_numpy(output), to_numpy(heat_maps), to_numpy(meta["iris_diameter"]) / 2.0)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            acces.update(acc[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_step == 0:
                print(
                    '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.2e} | Acc: {acc1: .3f}, {acc2: .3f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc1=acc[0],
                        acc2=acc[1]
                    ))
            bar.next()

        bar.finish()
    return losses.avg, acces.avg, best_acc


if __name__ == '__main__':

    EPOCH_SIZE = 5
    BATCH_SIZE = 4
    NUM_WORKERS = 1
    SNAPSHOT = 2
    DATA_FOLDER = r"data"
    CHECKPOINT_PATH = r"checkpoints\exp1"

    LR = 0.005

    # create checkpoint dir
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model creating
    NUM_STACKS = 3
    NUM_BLOCKS = 4
    NUM_CLASSES = 17
    print("==> creating model: stacks={}, blocks={}".format(NUM_STACKS, NUM_BLOCKS))
    model = HourglassNet(num_stacks=NUM_STACKS, num_blocks=NUM_BLOCKS, num_classes=NUM_CLASSES)
    model = model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = JointsMSELoss().to(device)
    optimizer = get_optimizer("adam", model, lr=LR)
    best_acc = 0.0

    LOAD_FILE = None  # None | r''

    start_epoch = 0
    if LOAD_FILE is not None:
        print("=> loading checkpoint '{}'".format(LOAD_FILE))
        checkpoint = torch.load(LOAD_FILE)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(LOAD_FILE, checkpoint['epoch']))

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # create data loader
    unity_eye = EyeLandmarksDataset(DATA_FOLDER, load_full=False)
    train_dataset = TrainDataset(unity_eye, data_config={
        "max_shift": (5, 7),
        "delta_scale": 0.4,
        "max_rotation_angle": 0.5,
        "image_size": (128, 96),
        "line_count": 2,
        "down_up_scale": 0.4,
        "sigma_head_map": 35.0,
    })

    test_dataset = TestDataset(unity_eye, {
        "line_count": 0,
        "image_size": (128, 96),
    })

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # train and eval
    lr = LR

    lrs = []
    train_losses = []
    valid_losses = []
    train_acc_list = []
    valid_acc_list = []

    difficult = np.linspace(0.15, 0.95, EPOCH_SIZE)
    SCHEDULE = [1, 2, 4]
    GAMMA = 0.8
    for epoch in range(start_epoch, EPOCH_SIZE):
        lr = adjust_learning_rate(optimizer, epoch, lr, SCHEDULE, GAMMA)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        train_dataset.set_difficult(difficult[epoch])
        test_dataset.set_difficult(difficult[epoch])
        # train for one epoch
        train_loss, train_acc, best_acc = train(train_dataloader, model, criterion, optimizer, train_size=800,
                                                print_step=1, save_step=2, checkpoint_path=CHECKPOINT_PATH,
                                                best_acc=best_acc)

        # evaluate on validation set
        valid_loss, valid_acc, best_acc = validate(test_dataloader, model, criterion, print_step=1, best_acc=best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': max(valid_acc, best_acc),
            'optimizer': optimizer.state_dict(),
        }, valid_acc > best_acc, checkpoint=CHECKPOINT_PATH, snapshot=SNAPSHOT)

    for points in [lrs, train_losses, valid_losses, train_acc_list, valid_acc_list]:
        x = np.arange(len(points))
        plt.plot(x, np.asarray(points))
    plt.legend([name for name in ['LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']])
    plt.grid(True)

    plt.savefig(os.path.join(CHECKPOINT_PATH, 'log.eps'), dpi=150)
