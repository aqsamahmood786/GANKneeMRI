import torch
import torch.nn as nn
from torchvision import models
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True).features# denseNet 161#densenet121(1024), densenet161, alexnet
        self.fc = nn.Linear(256, 1)#256

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.alexnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))#self.dropout(out)

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out
