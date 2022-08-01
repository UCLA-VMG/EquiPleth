import torch

class SNRLoss_dB_Signals(torch.nn.Module):
    def __init__(self, N=1024, pulse_band=[45/60., 180/60.]):
        super(SNRLoss_dB_Signals, self).__init__()
        self.N = N
        self.pulse_band = torch.tensor(pulse_band, dtype=torch.float32)


    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, Fs=30):
        device = outputs.device
        self.pulse_band = self.pulse_band.to(device)
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()
        N_samp = outputs.shape[-1]
        wind_sz = int(self.N/256)

        f = torch.linspace(0, Fs/2, int(self.N/2)+1, dtype=torch.float32).to(device)

        min_idx = torch.argmin(torch.abs(f - self.pulse_band[0]))
        max_idx = torch.argmin(torch.abs(f - self.pulse_band[1]))

        outputs = outputs.view(-1, N_samp)
        targets = targets.view(-1, N_samp)

        # Generate GT heart indices from GT signals
        Y = torch.fft.rfft(targets, n=self.N, dim=1, norm='forward')
        Y2 = torch.abs(Y)**2
        HRixs = torch.argmax(Y2[:,min_idx:max_idx],axis=1)+min_idx

        X = torch.fft.rfft(outputs, n=self.N, dim=1, norm='forward')

        P1 = torch.abs(X)**2

        # Calc SNR for each batch
        losses = torch.empty((X.shape[0],), dtype=torch.float32)
        for count, ref_idx in enumerate(HRixs):
            pulse_freq_amp = torch.sum(P1[count, ref_idx-wind_sz:ref_idx+wind_sz])
            other_avrg = (torch.sum(P1[count, min_idx:ref_idx-wind_sz])+torch.sum(P1[count, ref_idx+wind_sz:max_idx]))
            losses[count] = -10*torch.log10(pulse_freq_amp/(other_avrg+1e-7))
        losses.to(device)
        return torch.mean(losses)

class SNRLossOnPreComputedAndWindowedFFT_base(torch.nn.Module):
    def __init__(self, start_idx, window_fraction=0.02, device=torch.device('cpu')):
        super(SNRLossOnPreComputedAndWindowedFFT_base, self).__init__()
        self.start_idx = start_idx
        self.window_fraction = window_fraction
        self.device = device
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        assert outputs.shape == targets.shape, "The predicted output and the target labels have different shapes"
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()
        # The window length for the calculating the signal to noise ratio of the output's PSD
        # Here we take 5% in total, i.e. 2.5% on both sides
        window_size_base_harmonic = int(self.window_fraction * targets.shape[1]) + 1

        # Get the strongest peaks from the output, i.e. the heart beats from the PSD of the targets
        Y2 = torch.abs(targets) ** 2
        HRixs = torch.argmax(Y2,axis=1)

        # Get the PSD of the outputs of the neural network
        X2 = torch.abs(outputs) ** 2

        # Calc SNR for each batch
        losses = torch.empty((X2.shape[0],), dtype=torch.float32)
        for count, ref_idx in enumerate(HRixs):
            # Compute the power around the heart beat idx and its fist harmonic => Signal
            base_start = max([0, ref_idx - window_size_base_harmonic])
            base_end   = ref_idx + window_size_base_harmonic + 1
            pulse_freq_amp = torch.sum(X2[count, base_start : base_end])
            # Compute the power outisde the above windows => Noise
            other_avrg = torch.sum(X2[count, :base_start]) + torch.sum(X2[count, base_end:])
            # Take the SNR loss in decibels
            losses[count] = -10*torch.log10(pulse_freq_amp/(other_avrg+1e-7))
        losses.to(self.device)
        return torch.mean(losses)