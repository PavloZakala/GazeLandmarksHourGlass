import torch.nn as nn

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        loss = 0

        for idx in range(batch_size):
            # heatmap_output = output[:, idx]
            # heatmap_target = target[:, idx]
            heatmap_output = output[idx]
            heatmap_target = target[idx]
            loss += 0.5 * self.criterion(heatmap_output, heatmap_target)

        return loss / num_joints