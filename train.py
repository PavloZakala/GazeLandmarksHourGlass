import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.data_preprocessing import EyeLandmarksDataset, TrainDataset, TestDataset
from model.hourglass import make_hourglass
from utils.data_augmentation import get_landmarks_from_heatmap
from utils import metrics


def validation_model(model, dataloader):
    for image, head_map in dataloader:
        h, w, _ = image.shape
        target_landmarks = get_landmarks_from_heatmap(head_map)

        ac, cnt = metrics.pck_eval(target_landmarks, target_landmarks, image_size=(w, h))


if __name__ == '__main__':
    EPOCH_SIZE = 10
    BATCH_SIZE = 8
    NUM_WORKERS = 1
    DATA_FOLDER = r"/home/pavlo/PycharmProjects/GazeLandmarksHourGlass/data"

    unity_eye = EyeLandmarksDataset(DATA_FOLDER, load_full=False)
    train_dataset = TrainDataset(unity_eye, data_config={
        "max_shift": (5, 7),
        "delta_scale": 0.2,
        "max_rotation_angle": 0.5,
        "image_size": (120, 72),
        "line_count": 2,
        "down_up_scale": 0.4,
        "sigma_head_map": 35.0,
    })

    test_dataset = TestDataset(unity_eye, {
        "line_count": 2,
        "image_size": (120, 72),
    })

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = make_hourglass(num_stacks=3, num_blocks=4, num_classes=17)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCH_SIZE):
        for image, heat_map in train_dataloader:
            print(image.shape)

            # if torch.cuda.is_available():
