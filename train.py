import torch
import os
import time
import numpy as np
from progress.bar import Bar as Bar
from matplotlib import pyplot as plt

from utils.tools import adjust_learning_rate, save_checkpoint
from utils import transforms
from torch.utils.data import DataLoader
from utils.imutils import batch_with_heatmap
from model.hourglass import HourglassNet
from utils.evaluation import AverageMeter, accuracy, final_preds
from utils.data_preprocessing import EyeLandmarksDataset, TrainDataset, TestDataset
from losses import JointsMSELoss
from utils.optimizer import get_optimizer

best_acc = 0.0


def train(train_loader, model, criterion, optimizer, debug=False):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device, non_blocking=True)
        # target_weight = meta['target_weight'].to(device, non_blocking=True)

        # compute output
        output = model(input)
        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o, target)
            output = output[-1]
        else:  # single output
            loss = criterion(output, target)
        acc = accuracy(output, target, np.arange(17))

        if debug:  # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, output)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print(
            '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                batch=i + 1,
                size=len(train_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                acc=acces.avg
            ))
        bar.next()

    bar.finish()
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, num_classes=17, debug=False):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(input)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()

            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target)
                output = output[-1]
            else:  # single output
                loss = criterion(output, target)

            acc = accuracy(score_map, target.cpu(), np.arange(17))

            # generate predictions
            # preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:
                gt_batch_img = batch_with_heatmap(input, target)
                pred_batch_img = batch_with_heatmap(input, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            print(
                '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=acces.avg
                ))
            bar.next()

        bar.finish()
    return losses.avg, acces.avg, predictions


if __name__ == '__main__':

    EPOCH_SIZE = 10
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    SNAPSHOT = 2
    # DATA_FOLDER = r"/home/pavlo/PycharmProjects/GazeLandmarksHourGlass/data"
    DATA_FOLDER = r"C:\Users\Pavlo\PycharmProjects\GazeLandmarksHourGlass\data"
    CHECKPOINT_PATH = r"C:\Users\Pavlo\PycharmProjects\GazeLandmarksHourGlass\checkpoints\exp1"

    LR = 0.0001
    SCHEDULE = [3, 6, 9]
    GAMMA = 0.8

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
        "delta_scale": 0.2,
        "max_rotation_angle": 0.5,
        "image_size": (128, 96),
        "line_count": 2,
        "down_up_scale": 0.4,
        "sigma_head_map": 35.0,
    })

    test_dataset = TestDataset(unity_eye, {
        "line_count": 2,
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

    for epoch in range(start_epoch, EPOCH_SIZE):
        lr = adjust_learning_rate(optimizer, epoch, lr, SCHEDULE, GAMMA)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        train_dataset.set_difficult((epoch + 1) / EPOCH_SIZE)
        test_dataset.set_difficult((epoch + 1) / EPOCH_SIZE)
        # train for one epoch
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(test_dataloader, model, criterion)

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, predictions, is_best, checkpoint=CHECKPOINT_PATH, snapshot=SNAPSHOT)

    for points in [lrs, train_losses, valid_losses, train_acc_list, valid_acc_list]:
        x = np.arange(len(points))
        plt.plot(x, np.asarray(points))
    plt.legend([name for name in ['LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']])
    plt.grid(True)

    plt.savefig(os.path.join(CHECKPOINT_PATH, 'log.eps'), dpi=150)
