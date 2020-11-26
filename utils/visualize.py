import numpy as np
from matplotlib import pyplot as plt
import cv2


def show_image_with_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    implot = plt.imshow(image)
    plt.imshow(heatmap, alpha=.3)
    plt.show()


def show_image_with_landmarks(image, landmarks):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    for p in landmarks:
        p = p.astype(np.int)
        image = cv2.circle(image, (int(p[0]), int(p[1])), 1, (255, 0, 0), 2)

    plt.imshow(image)
    plt.show()
