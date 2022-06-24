import torch
import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        feats = torch.reshape(img, (img.shape[0], img.shape[2]*img.shape[3]))
        return feats