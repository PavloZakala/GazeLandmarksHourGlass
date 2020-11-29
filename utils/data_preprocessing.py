import glob
import os

import pickle as pkl
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.tools import load_landmarks, multivariate_gaussian
from utils import data_augmentation
from utils import metrics

# DATA_FOLDER = r"/home/pavlo/PycharmProjects/GazeLandmarksHourGlass/data"
DATA_FOLDER = r"data"

def _get_item(idx, dataset, configs:dict):
    image, landmarks = dataset[idx]

    MAX_SHIFT = configs.get("max_shift", None)
    DELTA_SCALE = configs.get("delta_scale", None)
    MAX_ROTATION_ANGLE = configs.get("max_rotation_angle", None)
    IMAGE_SIZE = configs.get("image_size", None)
    LINE_COUNT = configs.get("line_count", None)
    DOWN_UP_SCALE = configs.get("down_up_scale", None)
    SIGMA_HEAD_MAP = configs.get("sigma_head_map", None)
    difficult = configs.get("difficult", None)

    h, w, _ = image.shape

    x_borders = (w / 6, 5 * w / 6)
    y_borders = (h / 8, 7 * h / 8)

    data = {
        "image": image,
        "bound_box": (x_borders, y_borders),
        "landmarks": landmarks,
    }

    data_augmentation.shift(data, max_shift=MAX_SHIFT, difficult=difficult)
    data_augmentation.scale(data, delta_value=DELTA_SCALE, difficult=difficult)
    data_augmentation.random_rotation(data, max_angle=MAX_ROTATION_ANGLE, difficult=difficult)

    data_augmentation.crop(data)
    data_augmentation.resize(data, size=IMAGE_SIZE)

    data_augmentation.random_gamma_corrected(data, difficult=difficult)
    data_augmentation.add_line(data, count=LINE_COUNT, difficult=difficult)
    data_augmentation.down_up_scale(data, scale=DOWN_UP_SCALE, difficult=difficult)

    data_augmentation.make_map(data, size=(IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2),
                               sigma=SIGMA_HEAD_MAP, difficult=difficult)
    data_augmentation.data_normalizarion(data)

    iris_diameter = np.linalg.norm(data["landmarks"][[-2]] - data["landmarks"][6:14], axis=1).max()

    meta = {
        "landmarks": data["landmarks"],
        "iris_center": data["landmarks"][-2],
        "iris_diameter": iris_diameter,
    }

    return np.transpose(data["image"].astype(np.float32), (2, 0, 1)), data["heat_map"].astype(np.float32), meta

class TrainDataset(Dataset):

    def __init__(self, dataset, data_config={}):
        self._dataset = dataset
        self._indexes = dataset.get_train_indexes()

        self.difficult = 0.0  # [0.0, 1.0]

        self.MAX_SHIFT = data_config.get("max_shift", (5, 7))
        self.DELTA_SCALE = data_config.get("delta_scale", 0.2)
        self.MAX_ROTATION_ANGLE = data_config.get("max_rotation_angle", 0.3)
        self.IMAGE_SIZE = data_config.get("image_size", (120, 72))
        self.LINE_COUNT = data_config.get("line_count", 2)
        self.DOWN_UP_SCALE = data_config.get("down_up_scale", 0.4)
        self.SIGMA_HEAD_MAP = data_config.get("sigma_head_map", 35.0)

        print("***********************************")
        print("Max shift:", self.MAX_SHIFT)
        print("Delta scale:", self.DELTA_SCALE)
        print("Max rotation angle:", self.MAX_ROTATION_ANGLE)
        print("Image size:", self.IMAGE_SIZE)
        print("Line count:", self.LINE_COUNT)
        print("Down up scale:", self.DOWN_UP_SCALE)
        print("Sigma head map:", self.SIGMA_HEAD_MAP)
        print("***********************************")

    def set_difficult(self, difficult):
        self.difficult = difficult

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, item):
        res = _get_item(self._indexes[item], self._dataset,
                         {
                             "max_shift": self.MAX_SHIFT,
                             "delta_scale": self.DELTA_SCALE,
                             "max_rotation_angle": self.MAX_ROTATION_ANGLE,
                             "image_size": self.IMAGE_SIZE,
                             "line_count": self.LINE_COUNT,
                             "down_up_scale": self.DOWN_UP_SCALE,
                             "sigma_head_map":self.SIGMA_HEAD_MAP,
                             "difficult": self.difficult,
                         })
        return res

class TestDataset(Dataset):

    def __init__(self, dataset, data_config={}):
        self._dataset = dataset
        self._indexes = dataset.get_test_indexes()

        self.difficult = 0.0  # [0.0, 1.0]

        self.MAX_SHIFT = (3, 4)
        self.DELTA_SCALE = 0.0
        self.MAX_ROTATION_ANGLE = 0.3
        self.IMAGE_SIZE = data_config.get("image_size", (120, 72))
        self.LINE_COUNT = data_config.get("line_count", 2)
        self.DOWN_UP_SCALE = 0.0
        self.SIGMA_HEAD_MAP = 1.0
        print("***********************************")
        print("Max shift:", self.MAX_SHIFT)
        print("Delta scale:", self.DELTA_SCALE)
        print("Max rotation angle:", self.MAX_ROTATION_ANGLE)
        print("Image size:", self.IMAGE_SIZE)
        print("Line count:", self.LINE_COUNT)
        print("Down up scale:", self.DOWN_UP_SCALE)
        print("Sigma head map:", self.SIGMA_HEAD_MAP)
        print("***********************************")

    def set_difficult(self, difficult):
        self.difficult = difficult

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, item):
        res = _get_item(self._indexes[item], self._dataset,
                         {
                             "max_shift": self.MAX_SHIFT,
                             "delta_scale": self.DELTA_SCALE,
                             "max_rotation_angle": self.MAX_ROTATION_ANGLE,
                             "image_size": self.IMAGE_SIZE,
                             "line_count": self.LINE_COUNT,
                             "down_up_scale": self.DOWN_UP_SCALE,
                             "sigma_head_map": self.SIGMA_HEAD_MAP,
                             "difficult": self.difficult,
                         })
        return res

class EyeLandmarksDataset(Dataset):
    PATH = r"~/UnityEyes/imgs"
    FILE_SIZE = 5000

    @staticmethod
    def __data_preprocess(path):

        path_list = []
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            path_list = glob.glob(os.path.join(path, "*.plk"))
            return path_list

        image_list = glob.glob(os.path.join(EyeLandmarksDataset.PATH, "*.jpg"))

        images = []
        landmarks_list = []

        last = 1
        for i, img_path in enumerate(tqdm(image_list)):
            json_path = "{}.json".format(img_path[:-4])

            data = load_landmarks(json_path)
            image = cv2.imread(img_path)

            interior_margin_2d = np.array(data["interior_margin_2d"])[[1, 3, 5, 8, 11, 13, 15]][:, :2]
            iris_2d = np.array(data["iris_2d"])[[0, 4, 8, 12, 16, 20, 24, 28]][:, :2]
            iris_center = np.mean(iris_2d, axis=0).reshape(1, -1)[:, :2]
            eye_center = np.array(image.shape)[[1, 0, 2]].reshape(1, -1)[:, :2] / 2.0

            landmarks = np.concatenate([interior_margin_2d, iris_2d, iris_center, eye_center])
            landmarks[:, 1] = image.shape[0] - landmarks[:, 1]
            interior_margin_2d[:, 1] = image.shape[0] - interior_margin_2d[:, 1]

            x_min, y_min = interior_margin_2d.min(axis=0).astype(np.int)
            x_max, y_max = interior_margin_2d.max(axis=0).astype(np.int)
            xc = (x_min + x_max) // 2
            yc = (y_min + y_max) // 2

            h, w, _ = image.shape
            pre_x_borders = (xc - w / 6, xc + w / 6)
            pre_y_borders = (yc - h / 8, yc + h / 8)

            landmarks = landmarks - np.array([pre_x_borders[0], pre_y_borders[0]])
            image_crop = image[int(pre_y_borders[0]):int(pre_y_borders[1]),
                         int(pre_x_borders[0]):int(pre_x_borders[1])].copy()
            del image
            images.append(image_crop)
            landmarks_list.append(landmarks)

            if (i + 1) % EyeLandmarksDataset.FILE_SIZE == 0:
                with open(os.path.join(path, "unity_eyes_{}-{}.plk".format(last, i + 1)), "wb") as f:
                    pkl.dump({"images": images,
                              "landmarks": landmarks_list}, f)
                    path_list.append(os.path.join(path, "unity_eyes_{}-{}.plk".format(last, i + 1)))

                last = i + 1
                images = []
                landmarks_list = []

        with open(os.path.join(path, "unity_eyes_{}-{}.plk".format(last, i + 1)), "wb") as f:
            pkl.dump({"images": images,
                      "landmarks": landmarks_list}, f)
            path_list.append(os.path.join(path, "unity_eyes_{}-{}.plk".format(last, i + 1)))

        return path_list

    @staticmethod
    def __load_data(path_list, load_full=True):

        if not load_full:
            path_list = path_list[:1]

        images = []
        landmarks = []
        for path in tqdm(path_list):
            with open(path, "rb") as f:
                data = pkl.load(f)

            images.extend(data["images"])
            landmarks.extend(data["landmarks"])

        return images, landmarks

    def __init__(self, data_path, load_full=True):
        from sklearn.model_selection import train_test_split

        self._data_path = data_path
        self.path_list = self.__data_preprocess(self._data_path)
        self._images, self._landmarks = self.__load_data(self.path_list, load_full=load_full)
        indexes = np.arange(len(self._landmarks))
        self._train_indexes, self._test_indexes = train_test_split(indexes, train_size=0.85, random_state=0)

        self._indexes = None

    def get_train_indexes(self):
        return self._train_indexes

    def get_test_indexes(self):
        return self._test_indexes

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx):
        return self._images[idx], self._landmarks[idx]


if __name__ == '__main__':

    unity_eye = EyeLandmarksDataset(DATA_FOLDER, load_full=False)
    dataset = TestDataset(unity_eye, {
        "line_count": 0,
        "image_size": (128, 96),
    })
    dataset.set_difficult(0.5)
    difficult = 0.0
    from utils.visualize import show_image_with_heatmap, show_image_with_landmarks
    for idx in tqdm(range(len(dataset))):
        image, heat_map, meta = dataset[idx]
        show_image_with_heatmap(np.transpose(image, (1, 2, 0)), heat_map.sum(0))
        show_image_with_landmarks(np.transpose(image, (1, 2, 0)), meta["landmarks"])

        print(image.shape)
        print(heat_map.shape)
        image = (256 * np.transpose(image, (1, 2, 0))).astype(np.uint8)
        cv2.imshow("image", image)
        cv2.imshow("head_map", heat_map.sum(0))
        key = cv2.waitKey()

        if key == ord('q'):
            break
