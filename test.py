import os
import time

import numpy as np
import torch
from progress.bar import Bar as Bar
from torch.utils.data import DataLoader

from model.hourglass import HourglassNet
from utils import data_augmentation
from utils.data_augmentation import get_landmarks_from_heatmap
from utils.data_preprocessing import EyeLandmarksDataset, TestDataset
from utils.metrics import landmarks_metrics_eval, AverageMeter
from utils.tools import to_numpy
from utils.visualize import show_image_with_heatmap, show_image_with_landmarks

DEBUG = True
LOAD_FILE = r"C:\Users\Pavlo\PycharmProjects\GazeLandmarksHourGlass\checkpoints\exp2\model_best.pth"

COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
          (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
          (0, 255, 255), (255, 0, 255), (0, 255, 0)]


def get_model(device):
    NUM_STACKS = 3
    NUM_BLOCKS = 4
    NUM_CLASSES = 17
    print("==> creating model: stacks={}, blocks={}".format(NUM_STACKS, NUM_BLOCKS))
    model = HourglassNet(num_stacks=NUM_STACKS, num_blocks=NUM_BLOCKS, num_classes=NUM_CLASSES)
    model = model.to(device)

    print("=> loading checkpoint '{}'".format(LOAD_FILE))
    checkpoint = torch.load(LOAD_FILE, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    return model


def test_unity_eyes():
    DATA_FOLDER = r"C:\Users\Pavlo\PycharmProjects\GazeLandmarksHourGlass\data"
    PRINT_SIZE = 10
    SAVE_FOLDER = r"sources\unityeyes"
    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(device)

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
            if i == 10:
                break
            data_time.update(time.time() - end)

            images = images.to(device)
            heat_maps = heat_maps.to(device, non_blocking=True)

            # compute output
            output = model(images)[-1]
            if DEBUG:
                for image, target_heat_map, predict_heat_map in zip(to_numpy(images), to_numpy(heat_maps),
                        to_numpy(output)):
                    show_image_with_heatmap(np.transpose(image, (1, 2, 0)), target_heat_map.sum(0),
                                            save_name=os.path.join(SAVE_FOLDER,
                                                                   "unityeyes_{}_target_heatmap.jpg".format(i)))
                    show_image_with_heatmap(np.transpose(image, (1, 2, 0)), predict_heat_map.sum(0),
                                            save_name=os.path.join(SAVE_FOLDER,
                                                                   "unityeyes_{}_predict_heatmap.jpg".format(i)))

                    target_landmarks = get_landmarks_from_heatmap([target_heat_map])[0] * 2.0
                    predict_landmarks = get_landmarks_from_heatmap([predict_heat_map])[0] * 2.0

                    show_image_with_landmarks(np.transpose(image, (1, 2, 0)), target_landmarks, colors=COLORS,
                                              save_name=os.path.join(SAVE_FOLDER,
                                                                     "unityeyes_{}_target_landmarks.jpg".format(i)))
                    show_image_with_landmarks(np.transpose(image, (1, 2, 0)), predict_landmarks, colors=COLORS,
                                              save_name=os.path.join(SAVE_FOLDER,
                                                                     "unityeyes_{}_predict_landmarks.jpg".format(i)))

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


def test_images():
    import glob
    import cv2
    import os

    name = "cave"  # "dirl", "cave", "mpii"
    if not os.path.isdir(name):
        os.mkdir(name)

    if name == "dirl":
        image_paths = glob.glob(r"{}\*.png".format(r"C:\Users\Pavlo\Desktop\test_images\dirl"))
    elif name == "cave":
        image_paths = glob.glob(r"{}\*.jpg".format(r"C:\Users\Pavlo\Desktop\test_images\cave"))
    elif name == "mpii":
        image_paths = glob.glob(r"{}\*.jpg".format(r"C:\Users\Pavlo\Desktop\test_images\mpii"))
    else:
        raise NotImplemented()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)

    for i, img_path in enumerate(image_paths):

        images = cv2.imread(img_path)
        data = {"image": images}
        data_augmentation.resize(data, size=(128, 96))
        data_augmentation.data_normalizarion(data)
        images = np.transpose(data["image"].astype(np.float32), (2, 0, 1))[np.newaxis, :]

        images = torch.from_numpy(images).to(device)
        output = model(images)[-1]
        if DEBUG:
            for image, predict_heat_map in zip(to_numpy(images), to_numpy(output)):
                show_image_with_heatmap(np.transpose(image, (1, 2, 0)), predict_heat_map.sum(0),
                                        save_name=os.path.join(name, "{}_{}_heatmap.jpg".format(name, i)))

                predict_landmarks = get_landmarks_from_heatmap([predict_heat_map])[0] * 2.0

                show_image_with_landmarks(np.transpose(image, (1, 2, 0)), predict_landmarks, COLORS,
                                          save_name=os.path.join(name, "{}_{}_landmarks.jpg".format(name, i)))


if __name__ == '__main__':
    test_unity_eyes()
    # test_images()
