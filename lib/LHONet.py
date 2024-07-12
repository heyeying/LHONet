import logging
from torchvision.transforms.functional import rgb_to_grayscale
from lib.pvtv2 import pvt_v2_b2
from lib.LHONet_decoder import LHONet_Model
from lib.modules import *

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if (up.shape[2] != current.shape[2]
                or up.shape[3] != current.shape[3]):
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class LHONet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'     # Pre-training weight path
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # decoder initialization
        self.decoder = LHONet_Model(channels=[64, 128, 320, 512])

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(64, n_class, 1)
        self.out_head2 = nn.Conv2d(128, n_class, 1)
        self.out_head3 = nn.Conv2d(320, n_class, 1)
        self.out_head4 = nn.Conv2d(512, n_class, 1)

    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x2_o, x3_o = self.decoder(x1, x2, x3, x4, edge_feature)

        p2 = F.interpolate(x2_o, scale_factor=8, mode='bilinear')
        p3 = F.interpolate(x3_o, scale_factor=16, mode='bilinear')
        return p2, p3


if __name__ == '__main__':
    model = LHONet()
    input_tensor = torch.randn(1, 3, 352, 352)

    p2, p3 = model(input_tensor)
    print(p2.size(), p3.size())

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    print(flop_count_table(flops))
    print(model)

'''
| module                                      | #parameters or shape   | #flops     |
|:--------------------------------------------|:-----------------------|:-----------|
| model                                       | 28.066M                | 12.718G    |
'''