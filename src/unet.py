import torch

import math

def sinusoidal_embedding(t, dim):
    device = t.device
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / (half - 1))
    args = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)
    
class SafeDown(Down):
    def forward(self, x):
        if min(x.shape[-2:]) > 2:
            self.pool(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512, 1024), num_classes=1):
        super().__init__()
        
        self.encs = torch.nn.ModuleList([DoubleConv(in_channels, channels[0]),])
        self.downs = torch.nn.ModuleList([SafeDown(),])

        for i in range(len(channels) - 1):
            self.encs.append(DoubleConv(channels[i], channels[i+1]))
            self.downs.append(SafeDown())        

        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        self.ups = torch.nn.ModuleList()
        self.decs = torch.nn.ModuleList()

        for i in range(len(channels)-1, 0, -1):
            self.ups.append(Up(channels[i], channels[i - 1]))
            self.decs.append(DoubleConv(channels[i], channels[i - 1]))

        self.up0 = Up(channels[0], channels[0] // 2)
        self.dec0 = torch.nn.Conv2d(channels[0] // 2, channels[0] // 2, 1)
        self.outc = torch.nn.Conv2d(channels[0] // 2, num_classes, 1)

        # alt ending if predicting same level info
        self.dec0_same = torch.nn.Conv2d(channels[0], channels[0] // 2, 1)
        self.outc_same = self.outc = torch.nn.Conv2d(channels[0] // 2, num_classes, 1)

        # time embeddings
        self.sinusoidal_embedding = sinusoidal_embedding
        time_dim = 256

        # Encoder time MLPs
        self.time_mlp_enc = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(time_dim, channels[i]),
                torch.nn.ReLU(),
                torch.nn.Linear(channels[i], channels[i]),
            )
            for i in range(len(channels)-1)
        ])

        # Bottleneck time MLP
        self.time_mlp_bottom = torch.nn.Sequential(
            torch.nn.Linear(time_dim, channels[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[-1], channels[-1])
        )

        # Decoder time MLPs
        self.time_mlp_dec = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(time_dim, channels[i]),
                torch.nn.ReLU(),
                torch.nn.Linear(channels[i], channels[i])
            )
            for i in range(len(channels)-2, -1, -1)
        ])

    # TODO: @Ethan add positional encoding
    def forward(self, x):
        xs = []

        for i, enc in enumerate(self.encs[:-1]):
            x = enc(x)
            xs.append(x)
            x = self.downs[i](x)
            
        x = self.bottleneck(x)

        for i, dec in enumerate(self.decs):
            
            x_i = xs[::-1][i]
            x = self.ups[i](x)

            diffY = x_i.shape[2] - x.shape[2]
            diffX = x_i.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([x_i, x], dim=1)
            x = dec(x)
            
        x = self.up0(x)
        x = self.dec0(x)
        x = self.outc(x)

        return x
    

    def forwardt(self, t, x):
        temb_NT = self.sinusoidal_embedding(t, 256) 
        xs = []

        for i, enc in enumerate(self.encs[:-1]):
            x = enc(x)

            tproj = self.time_mlp_enc[i](temb_NT).view(x.shape[0], -1, 1, 1)
            x = x + tproj

            xs.append(x)
            x = self.downs[i](x)
            
        x = self.bottleneck(x)
        tproj = self.time_mlp_bottom(temb_NT).view(x.shape[0], -1, 1, 1)
        x = x + tproj

        for i, dec in enumerate(self.decs):
            
            x_i = xs[::-1][i]
            x = self.ups[i](x)

            diffY = x_i.shape[2] - x.shape[2]
            diffX = x_i.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([x_i, x], dim=1)
            x = dec(x)

            tproj = self.time_mlp_dec[i](temb_NT).view(x.shape[0], -1, 1, 1)
            x = x + tproj
            
        x = self.up0(x)
        x = self.dec0(x)
        x = self.outc(x)

        return x

    
    def forwardt_same(self, t, x): 
        # shares most with forward
        temb_NT = self.sinusoidal_embedding(t, 256) 
        xs = []

        for i, enc in enumerate(self.encs[:-1]):
            x = enc(x)

            tproj = self.time_mlp_enc[i](temb_NT).view(x.shape[0], -1, 1, 1)
            x = x + tproj

            xs.append(x)
            x = self.downs[i](x)
            
        x = self.bottleneck(x)
        tproj = self.time_mlp_bottom(temb_NT).view(x.shape[0], -1, 1, 1)
        x = x + tproj

        for i, dec in enumerate(self.decs):
            
            x_i = xs[::-1][i]
            x = self.ups[i](x)

            diffY = x_i.shape[2] - x.shape[2]
            diffX = x_i.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([x_i, x], dim=1)
            x = dec(x)

            tproj = self.time_mlp_dec[i](temb_NT).view(x.shape[0], -1, 1, 1)
            x = x + tproj
            
        x = self.dec0_same(x)
        x = self.outc_same(x)

        return x
