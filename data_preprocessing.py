import glob
import os

import pickle as pkl
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import data_augmentation
from utils import load_landmarks, multivariate_gaussian

DATA_FOLDER = r"/home/pavlo/PycharmProjects/gaze/data"


class EyeLandmarksDataset(Dataset):
    PATH = r"/home/pavlo/UnityEyes/imgs"
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
    def make_map(data, sigma=1.0, size=(120, 72)):

        heat_map = []
        for point in data["landmarks"]:
            X = np.arange(size[0]).astype(np.float32)
            Y = np.arange(size[1]).astype(np.float32)
            X, Y = np.meshgrid(X, Y)

            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y

            Z = multivariate_gaussian(pos, point, np.array([[sigma, 0.], [0., sigma]]))
            heat_map.append(Z / Z.max())
        data["heat_map"] = np.stack(heat_map)

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

    def __init__(self, data_path, load_full=True, test=False, data_config={}):
        from sklearn.model_selection import train_test_split

        self._data_path = data_path
        self.path_list = self.__data_preprocess(self._data_path)
        self._images, self._landmarks = self.__load_data(self.path_list, load_full=load_full)
        indexes = np.arange(len(self._images))
        train_indexes, test_indexes = train_test_split(indexes, train_size=0.85, random_state=0)

        self.difficult = 0.0  # [0.0, 1.0]
        self._indexes = None
        if test:
            self._indexes = test_indexes

            self.MAX_SHIFT = (3, 4)
            self.DELTA_SCALE = 0.0
            self.MAX_ROTATION_ANGLE = 0.3
            self.IMAGE_SIZE = data_config.get("image_size", (120, 72))
            self.LINE_COUNT = data_config.get("line_count", 2)
            self.DOWN_UP_SCALE = 0.0
            self.SIGMA_HEAD_MAP = 1.0
        else:
            self._indexes = train_indexes

            self.MAX_SHIFT = data_config.get("max_shift", (5, 7))
            self.DELTA_SCALE = data_config.get("delta_scale", 0.2)
            self.MAX_ROTATION_ANGLE = data_config.get("max_rotation_angle", 0.3)
            self.IMAGE_SIZE = data_config.get("image_size", (120, 72))
            self.LINE_COUNT = data_config.get("line_count", 2)
            self.DOWN_UP_SCALE = data_config.get("down_up_scale", 0.4)
            self.SIGMA_HEAD_MAP = data_config.get("sigma_head_map", 35.0)

        print("Max shift:", self.MAX_SHIFT)
        print("Delta scale:", self.DELTA_SCALE)
        print("Max rotation angle:", self.MAX_ROTATION_ANGLE)
        print("Image size:", self.IMAGE_SIZE)
        print("Line count:", self.LINE_COUNT)
        print("Down up scale:", self.DOWN_UP_SCALE)
        print("Sigma head map:", self.SIGMA_HEAD_MAP)

    def set_difficult(self, difficult):
        self.difficult = difficult

    def get_indexes(self):
        return self._indexes

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx):
        image = self._images[idx]
        landmarks = self._landmarks[idx]

        h, w, _ = image.shape

        x_borders = (w / 6, 5 * w / 6)
        y_borders = (h / 8, 7 * h / 8)

        data = {
            "image": image,
            "bound_box": (x_borders, y_borders),
            "landmarks": landmarks,
        }

        data_augmentation.shift(data, max_shift=self.MAX_SHIFT, difficult=self.difficult)
        data_augmentation.scale(data, delta_value=self.DELTA_SCALE, difficult=self.difficult)
        data_augmentation.random_rotation(data, max_angle=self.MAX_ROTATION_ANGLE, difficult=self.difficult)

        data_augmentation.crop(data)
        data_augmentation.resize(data, size=self.IMAGE_SIZE)

        data_augmentation.random_gamma_corrected(data, difficult=self.difficult)
        data_augmentation.add_line(data, count=self.LINE_COUNT, difficult=self.difficult)
        data_augmentation.down_up_scale(data, scale=self.DOWN_UP_SCALE, difficult=self.difficult)

        data_augmentation.make_map(data, size=self.IMAGE_SIZE, sigma=self.SIGMA_HEAD_MAP, difficult=self.difficult)

        return data["image"], data["heat_map"]


if __name__ == '__main__':

    dataset = EyeLandmarksDataset(DATA_FOLDER, load_full=False, data_config={
        "max_shift": (5, 7),
        "delta_scale": 0.2,
        "max_rotation_angle": 0.5,
        "image_size": (120, 72),
        "line_count": 2,
        "down_up_scale": 0.4,
        "sigma_head_map": 35.0,
    })
    dataset.set_difficult(0.6)
    difficult = 0.0
    for idx in tqdm(dataset.get_indexes()):
        image, heat_map = dataset[idx]

        cv2.imshow("image", cv2.resize(image, (120 * 3, 72 * 3)))
        cv2.imshow("head_map", cv2.resize(heat_map.sum(0), (120 * 4, 72 * 4)))
        key = cv2.waitKey()

        if key == ord('q'):
            break
