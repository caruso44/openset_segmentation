import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class Doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)



class UNET(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 1, feat = False, features = [64,128,256,512]):
        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #Down part of UNET
        for feature in features:
            self.down.append(Doubleconv(in_channel, feature))
            in_channel = feature
        
        #Upper part of UNET
        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(feature * 2, feature, 2, 2)
            )
            self.up.append(Doubleconv(feature*2, feature))
        
        self.bottleneck = Doubleconv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size = 1)
        self.feat = feat
    
    def forward(self, x):
        skip_connection = []

        for down in self.down:
            x = down(x)
            skip_connection.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1]
        y = 0
        for idx in range(0,len(self.up),2):
            y = x
            x = self.up[idx](x)
            skip = skip_connection[idx//2]
            if x.shape != skip.shape:
                F.resize(x,size=skip.shape[2:])
            conc_skip = torch.concat((skip,x), dim=1) 
            x = self.up[idx + 1](conc_skip)
        if self.feat == True:
            return self.final_conv(x), x, nn.functional.interpolate(y, x.size()[2:], mode='bilinear')
        return self.final_conv(x)