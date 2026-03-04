import torch.nn as nn

def cba(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )

class YOLOModel(nn.Module):
    def __init__(self, classes: int, grid_size: int, boxes: int):
        super().__init__()
        self.C = classes
        self.S = grid_size
        self.B = boxes
        self.cell_out = self.B * 5 + self.C

        self.backbone = nn.Sequential(
            cba(3, 32),
            nn.MaxPool2d(2),
            cba(32, 64),
            nn.MaxPool2d(2),
            cba(64, 128),
            nn.MaxPool2d(2),
            cba(128, 256),
            nn.MaxPool2d(2),
            cba(256, 512),
            nn.AdaptiveAvgPool2d((self.S, self.S)),
        )

        self.head = nn.Conv2d(512, self.cell_out, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)  # (N, 512, S, S)
        pred = self.head(feat)  # (N, cell_out, S, S)
        pred = pred.permute(0, 2, 3, 1)  # (N, S, S, cell_out)
        return pred
