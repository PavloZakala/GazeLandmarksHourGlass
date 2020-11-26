import time
import torch
import numpy as np
from progress.bar import Bar as Bar
from torch.utils.data import DataLoader

from utils.metrics import landmarks_metrics_eval, AverageMeter
from model.hourglass import HourglassNet
from utils.data_preprocessing import EyeLandmarksDataset, TrainDataset, TestDataset
from utils.tools import adjust_learning_rate, save_checkpoint, to_numpy
from utils.visualize import show_image_with_heatmap, show_image_with_landmarks
from utils.data_augmentation import get_landmarks_from_heatmap

DEBUG = True

if __name__ == '__main__':
    LOAD_FILE = r"C:\Users\Pavlo\PycharmProjects\GazeLandmarksHourGlass\checkpoints\exp1\model_best.pth.tar"
    DATA_FOLDER = r"C:\Users\Pavlo\PycharmProjects\GazeLandmarksHourGlass\data"
    PRINT_SIZE = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model creating
    NUM_STACKS = 1
    NUM_BLOCKS = 4
    NUM_CLASSES = 17
    print("==> creating model: stacks={}, blocks={}".format(NUM_STACKS, NUM_BLOCKS))
    model = HourglassNet(num_stacks=NUM_STACKS, num_blocks=NUM_BLOCKS, num_classes=NUM_CLASSES)
    model = model.to(device)

    print("=> loading checkpoint '{}'".format(LOAD_FILE))
    checkpoint = torch.load(LOAD_FILE)
    model.load_state_dict(checkpoint['state_dict'])

    unity_eye = EyeLandmarksDataset(DATA_FOLDER, load_full=False)
    test_dataset = TestDataset(unity_eye, {
        "line_count": 0,
        "image_size": (128, 96),
    })

    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    acces = AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    end = time.time()
    bar = Bar('Eval ', max=len(test_dataloader))
    with torch.no_grad():
        for i, (images, heat_maps, meta) in enumerate(test_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.to(device)
            heat_maps = heat_maps.to(device, non_blocking=True)

            # compute output
            output = model(images)[-1]
            if DEBUG:
                for image, target_heat_map, predict_heat_map in zip(to_numpy(images), to_numpy(heat_maps), to_numpy(output)):
                    show_image_with_heatmap(np.transpose(image, (1, 2, 0)), target_heat_map.sum(0))
                    show_image_with_heatmap(np.transpose(image, (1, 2, 0)), predict_heat_map.sum(0))

                    target_landmarks = get_landmarks_from_heatmap([target_heat_map])[0] * 2.0
                    predict_landmarks = get_landmarks_from_heatmap([predict_heat_map])[0] * 2.0

                    show_image_with_landmarks(np.transpose(image, (1, 2, 0)), target_landmarks)
                    show_image_with_landmarks(np.transpose(image, (1, 2, 0)), predict_landmarks)


            acc = landmarks_metrics_eval(to_numpy(output), to_numpy(heat_maps), to_numpy(meta["iris_diameter"]))

            acces.update(acc[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % PRINT_SIZE == 0:
                print(
                    '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |  Acc: {acc1: .3f}, {acc2: .3f}'.format(
                        batch=i + 1,
                        size=len(test_dataloader),
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        acc1=acc[0],
                        acc2=acc[1]
                    ))
            bar.next()

        bar.finish()