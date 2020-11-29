import numpy as np
from matplotlib import pyplot as plt
import cv2


def show_image_with_heatmap(image, heatmap, save_name=r""):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    implot = plt.imshow(image)
    plt.imshow(heatmap, alpha=.3)
    plt.savefig(save_name)
    image = cv2.imread(save_name)
    cv2.imwrite(save_name, image[67:-52, 86:-65])
    # plt.show()


def show_image_with_landmarks(image, landmarks, colors, save_name=r""):
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy() * 256.0).astype(np.uint8)

    for p, c in zip(landmarks, colors):
        p = p.astype(np.int)
        image = cv2.circle(image, (int(p[0]), int(p[1])), 1, c, 2)

    plt.imshow(image)
    plt.savefig(save_name)
    image = cv2.imread(save_name)
    cv2.imwrite(save_name, image[67:-52, 86:-65])
    # plt.show()
