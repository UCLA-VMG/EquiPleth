import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch

class Discriminator(nn.Module):
    def __init__(self, frames=64):  
        '''
        Multimodal PhysNet with modality fusion.
        '''
        super(Discriminator, self).__init__()

        self.FullBlock1 = nn.Sequential(
            nn.Linear(frames, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.FullBlock2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.FullBlock3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
        )
        self.FullBlock4 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
        )
        self.FullBlock5 = nn.Sequential(
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=True),
        )
        self.FullBlock6 = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):

        x = self.FullBlock1(x)
        x = self.FullBlock2(x)
        x = self.FullBlock3(x)
        x = self.FullBlock4(x)
        x = self.FullBlock5(x)
        x = self.FullBlock6(x)
        return x

class FusionModel(nn.Module):
    def __init__(self, base_ppg_est_len, rf_ppg_est_len, out_len, latent=512):
        super(FusionModel, self).__init__()
        self.base_ppg_est_len = base_ppg_est_len
        self.rf_ppg_est_len = rf_ppg_est_len
        self.latent = latent
        self.out_len = out_len
        self.Branch1Dense1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.Branch1Dense2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.Branch1Dense3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.Branch1Dense4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.Branch2Dense1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.Branch2Dense2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.Branch2Dense3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.Branch2Dense4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.FusionDense1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.FusionDense2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.FusionDense3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.FusionDense4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.FinalLayer   = nn.Conv1d(128, 1, kernel_size=1, stride=1)
    def forward(self, fft_ppg, fft_rf):
        fft_ppg = self.Branch1Dense1(fft_ppg)
        fft_ppg = self.Branch1Dense2(fft_ppg)
        fft_ppg = self.Branch1Dense3(fft_ppg)
        fft_ppg = self.Branch1Dense4(fft_ppg)
        fft_rf  = self.Branch2Dense1(fft_rf)
        fft_rf  = self.Branch2Dense2(fft_rf)
        fft_rf  = self.Branch2Dense3(fft_rf)
        fft_rf  = self.Branch2Dense4(fft_rf)
        fused_signal = torch.add(fft_ppg, fft_rf)
        fused_signal = self.FusionDense1(fused_signal)
        fused_signal = self.FusionDense2(fused_signal)
        fused_signal = self.FusionDense3(fused_signal)
        fused_signal = self.FusionDense4(fused_signal)
        output_signal = self.FinalLayer(fused_signal)
        return torch.squeeze(output_signal, 1)