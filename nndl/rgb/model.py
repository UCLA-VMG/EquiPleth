from torch import nn
class CNN3D(nn.Module):
    def __init__(self, frames=64, channels=3):  
        super(CNN3D, self).__init__()
        
        # Frames needs to be divisible by 4 due to the design specs.
        assert frames % 4 == 0
        
        # Convolutional layers.
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(channels, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        # Upsampling layers.
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        
        # Pooling layers.
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):                       	    	# [3, T, 128, 128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape
          
        x = self.ConvBlock1(x)		                        # [3, T, 128, 128]
        x = self.MaxpoolSpa(x)                              # [16, T, 64, 64]

        x = self.ConvBlock2(x)		                        # [32, T, 64, 64]
        x_visual6464 = self.ConvBlock3(x)	    	        # [32, T, 64, 64]
        x = self.MaxpoolSpaTem(x_visual6464)                # [32, T/2, 32, 32]

        x = self.ConvBlock4(x)		                        # [64, T/2, 32, 32]
        x_visual3232 = self.ConvBlock5(x)	    	        # [64, T/2, 32, 32]
        x = self.MaxpoolSpaTem(x_visual3232)                # [64, T/4, 16, 16]

        x = self.ConvBlock6(x)		                        # [64, T/4, 16, 16]
        x_visual1616 = self.ConvBlock7(x)	    	        # [64, T/4, 16, 16]
        x = self.MaxpoolSpa(x_visual1616)                   # [64, T/4, 8, 8]

        x = self.ConvBlock8(x)		                        # [64, T/4, 8, 8]
        x = self.ConvBlock9(x)		                        # [64, T/4, 8, 8]
        x = self.upsample(x)		                        # [64, T/2, 8, 8]
        x = self.upsample2(x)		                        # [64, T, 8, 8]
        
        x = self.poolspa(x)                                 # [64, T, 1,1]
        x = self.ConvBlock10(x)                             # [1, T, 1,1]
        
        rPPG = x.view(-1,length)                            # [T]

        return rPPG, x_visual, x_visual3232, x_visual1616