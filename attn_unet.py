import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, g, skip, inter):
        super(Attention, self).__init__()
        self.W_g = nn.Conv2d(g, inter, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(skip, inter, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        return x * psi
    
class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([x, y], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNet, self).__init__()
        self.encoder1 = Encoder(in_c, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)
        self.bottleneck = Encoder(512, 1024)
        self.decoder4 = Decoder(1024, 512)
        self.attention4 = Attention(512, 512, 256)
        self.decoder3 = Decoder(512, 256)
        self.attention3 = Attention(256, 256, 128)
        self.decoder2 = Decoder(256, 128)
        self.attention2 = Attention(128, 128, 64)
        self.decoder1 = Decoder(128, 64)
        self.attention1 = Attention(64, 64, 32)
        self.out = nn.Conv2d(64, out_c, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        x4 = self.decoder4.up(b)
        att4 = self.attention4(x4, e4)
        d4 = self.decoder4.conv(torch.cat([x4, att4], dim=1))

        x3 = self.decoder3.up(d4)
        att3 = self.attention3(x3, e3)
        d3 = self.decoder3.conv(torch.cat([x3, att3], dim=1))

        x2 = self.decoder2.up(d3)
        att2 = self.attention2(x2, e2)
        d2 = self.decoder2.conv(torch.cat([x2, att2], dim=1))

        x1 = self.decoder1.up(d2)
        att1 = self.attention1(x1, e1)
        d1 = self.decoder1.conv(torch.cat([x1, att1], dim=1))
        return self.out(d1)
