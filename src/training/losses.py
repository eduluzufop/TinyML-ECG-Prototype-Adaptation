from torch import nn


def get_classification_loss() -> nn.Module:
    return nn.CrossEntropyLoss()
