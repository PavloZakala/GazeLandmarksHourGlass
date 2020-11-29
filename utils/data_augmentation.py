import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.tools import multivariate_gaussian


def shift(data, max_shift=(5, 7), difficult=1.0):

    x_shift, y_shift = np.around((np.random.rand(2) * 2 - 1.0) * np.array(max_shift)) * difficult

    ((x_min, x_max), (y_min, y_max)) = data["bound_box"]
    data["bound_box"] = ((x_min + x_shift, x_max + x_shift), (y_min + y_shift, y_max + y_shift))


def scale(data, delta_value=0.1, difficult=1.0):

    score, = (np.random.rand(1) * 2 - 1.0) * delta_value * difficult + 1.0
    ((x_min, x_max), (y_min, y_max)) = data["bound_box"]

    xc = (x_max + x_min) / 2.0
    yc = (y_max + y_min) / 2.0

    xd = (x_max - x_min) * score
    yd = (y_max - y_min) * score

    data["bound_box"] = ((xc - xd / 2.0, xc + xd / 2.0),
                         (yc - yd / 2.0, yc + yd / 2.0))


def random_gamma_corrected(data, gamma_range=(0.5, 2.0), difficult=1.0):
    gamma_min, gamma_max = np.exp2(np.log2(gamma_range) * difficult)

    gamma, = np.random.rand(1) * (gamma_max - gamma_min) + gamma_min
    if "image" in data:
        data["image"] = np.array(255 * (data["image"] / 255) ** gamma, dtype='uint8')


def random_rotation(data, max_angle=0.3, difficult=1.0):

    angle, = (np.random.rand(1) * 2 - 1.0) * max_angle * difficult
    if "image" in data:
        image = data["image"]
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, np.degrees(angle), 1.0)
        data["image"] = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        if "landmarks" in data:
            ones = np.ones((data["landmarks"].shape[0], 1))
            data["landmarks"] = (rot_mat @ np.concatenate((data["landmarks"], ones), axis=1).T).T


def random_blur(data, sigma=1.0, difficult=1.0):
    if "image" in data:
        data["image"] = cv2.GaussianBlur(data["image"], (7, 7), sigmaX=sigma * difficult, sigmaY=sigma,
                                         borderType=cv2.BORDER_DEFAULT)


def down_up_scale(data, scale=0.4, difficult=1.0):
    if "image" in data:
        up_shape = np.array(data["image"].shape)[[1, 0]]
        scale = 1.0 - (1.0 - scale) * difficult
        down_shape = up_shape * scale
        down_image = cv2.resize(data["image"], tuple(down_shape.astype(np.int)))
        data["image"] = cv2.resize(down_image, tuple(up_shape.astype(np.int)))


def add_line(data, count=1, difficult=1.0):
    count = int(np.around(count * difficult))

    if "image" in data:
        shape = data["image"].shape
        for i in range(count):
            start = np.random.randint(-shape[1], 0), np.random.randint(-shape[0] // 2, (3 * shape[0]) // 2)
            end = np.random.randint(shape[1], 2 * shape[1]), np.random.randint(-shape[0] // 2, (3 * shape[0]) // 2)

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            thickness = np.random.randint(2, 3)
            data["image"] = cv2.line(data["image"], start, end, color, thickness)


def make_map(data, sigma=1.0, size=(120, 72), difficult=1.0):
    difficult = (difficult - 0.5) * 0.9 + 0.5

    sigma = sigma * (1.0 - (1.0 - (1.0 - difficult) ** 2.0) ** 0.5)
    heat_map = []
    for point in data["landmarks"]:
        X = np.arange(size[0]).astype(np.float32)
        Y = np.arange(size[1]).astype(np.float32)
        X, Y = np.meshgrid(X, Y)

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        Z = multivariate_gaussian(pos, point / 2.0, np.array([[sigma, 0.], [0., sigma]]))
        heat_map.append(Z / Z.max())
    data["heat_map"] = np.stack(heat_map)
    # plt.imshow(data["heat_map"].sum(0))
    # plt.show()


def crop(data):
    ((x_min, x_max), (y_min, y_max)) = data["bound_box"]

    if "landmarks" in data:
        data["landmarks"] = data["landmarks"] - np.array([x_min, y_min])

    if "image" in data:
        h, w, _ = data["image"].shape
        data["image"] = data["image"][min(0, int(y_min)):max(h, int(y_max)),
                                      min(0, int(x_min)):max(w, int(x_max))]


def resize(data, size=(120, 72)):
    if "image" in data:
        d_shape = np.array(size) / np.array(data["image"].shape)[[1, 0]]
        data["image"] = cv2.resize(data["image"], size)

        if "landmarks" in data:
            data["landmarks"] = data["landmarks"] * d_shape


def data_normalizarion(data):
    if "image" in data:
        data["image"] = data["image"] / 256.0


def get_landmarks_from_heatmap(batch_heat_map):
    return np.array([[np.unravel_index(np.argmax(map), map.shape) for map in heat_map]
                     for heat_map in batch_heat_map])[:, :, [1, 0]].astype(np.float32)

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals