import numpy as np
from utils.data_augmentation import get_landmarks_from_heatmap

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_dists_acc(preds, target, threshhold):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)

    dists = np.linalg.norm(preds - target, axis=2)
    return np.maximum((threshhold[:, np.newaxis] - dists) / threshhold[:, np.newaxis], 0.0)


def landmarks_metrics_eval(pred_heat_map, target_heat_map, threshhold):
    pred_landmarks = get_landmarks_from_heatmap(pred_heat_map)
    target_landmarks = get_landmarks_from_heatmap(target_heat_map)

    dists = calc_dists_acc(pred_landmarks, target_landmarks, threshhold=threshhold)
    bin_score = dists != 0.0

    return dists.mean(0).mean(0), bin_score.astype(np.float).mean()


if __name__ == '__main__':
    target = np.random.randint(0, 36, size=(5, 17, 2))
    predict = np.random.randint(0, 36, size=(5, 17, 2))

    dist = calc_dists_acc(predict, target, 15.0)
