
import numpy as np

import torch



def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        pred = outputs.max(1, keepdim=False)[1]
        return pred




def detect_rate(targets, pred):
    attack_id = np.where(targets.cpu() != pred.cpu())[0]
    return attack_id


def common(targets, pred):
    common_id = np.where(targets.cpu() == pred.cpu())[0]
    return common_id


def attack_suc(targets, pred):
    attack_id = np.where(targets.cpu() != pred.cpu())[0]
    return attack_id