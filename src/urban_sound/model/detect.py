import torch.nn as nn
import torch as th
from torchvision.models import squeezenet1_1


class ResnetDetector(nn.Module):
    def __init__(self, config):
        super(ResnetDetector, self).__init__()
        self.config = config
        self.device = self.config.device
        self.num_classes = self.config.dataset.num_classes
        self.model = squeezenet1_1(
            pretrained=False, num_classes=self.config.dataset.num_classes
        )
        self.model = self.model.to(self.device)

    def forward(self, batch):
        batch = batch.float().to(self.device)
        return self.model(batch)


class SimpleDetector(nn.Module):
    def __init__(self, config):
        super(SimpleDetector, self).__init__()
        self.config = config
        self.device = self.config.device
        param_value = th.randn(2)
        param_value = param_value / param_value.sum()
        self.register_parameter(name="param", param=nn.Parameter(param_value))

    def forward(self, batch):
        ret = th.ones(size=(batch.size(0), 2)) * (self.param / self.param.sum())
        return ret.to(self.device)


DETECT_MODEL_REGISTRY = {"simple": SimpleDetector, "resnet": ResnetDetector}


def get_model(config):
    return DETECT_MODEL_REGISTRY[config.detect.model_name](config)
