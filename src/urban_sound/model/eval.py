from torchtyping import TensorType
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super(SimpleClassifier, self).__init__()
        # self.size = 64
        # self.linear1 = nn.Linear(in_features=in_features, out_features=self.size)
        self.linear = nn.Linear(in_features=in_features, out_features=n_classes)
        # self.linear2 = nn.Linear(in_features=self.size, out_features=n_classes)

    def forward(self, batch: TensorType["batch", "in_features"]):
        # x = F.relu(self.linear1(batch))
        # return F.softmax(self.linear2(x))
        return F.softmax(self.linear(batch))
