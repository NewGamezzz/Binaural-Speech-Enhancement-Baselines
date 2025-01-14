import torch.nn as nn
from .shared import snr_loss, ild_loss_db, ipd_loss_rads, Stft, IStft
from torch_stoi import NegSTOILoss

class BinauralLoss(nn.Module):
    def __init__(self, win_len=400,
                 win_inc=100, fft_len=512, sr=16000,
                 ild_weight=0.1, ipd_weight=1, stoi_weight=0, 
                  snr_loss_weight=1):

        super().__init__()
        self.stft = Stft(fft_len, win_inc, win_len)
        self.istft = IStft(fft_len, win_inc, win_len)
        self.stoi_loss = NegSTOILoss(sample_rate=sr)
       
        self.ild_weight = ild_weight
        self.ipd_weight = ipd_weight
        self.stoi_weight = stoi_weight
        self.snr_loss_weight = snr_loss_weight

        
    def forward(self, model_output, targets):
        target_stft_l = self.stft(targets[:, 0])
        target_stft_r = self.stft(targets[:, 1])
        

        output_stft_l = self.stft(model_output[:, 0])
        output_stft_r = self.stft(model_output[:, 1])


        loss = 0
        if self.snr_loss_weight > 0:
            
            snr_l = snr_loss(model_output[:, 0], targets[:, 0])
            snr_r = snr_loss(model_output[:, 1], targets[:, 1])
          
            snr_loss_lr = - (snr_l + snr_r)/2
           
            bin_snr_loss = self.snr_loss_weight*snr_loss_lr
            
            print('\n SNR Loss = ', bin_snr_loss)
            loss += bin_snr_loss
        
        if self.stoi_weight > 0:
            stoi_l = self.stoi_loss(model_output[:, 0], targets[:, 0])
            stoi_r = self.stoi_loss(model_output[:, 1], targets[:, 1])

            stoi_loss = (stoi_l+stoi_r)/2
            bin_stoi_loss = self.stoi_weight*stoi_loss.mean()
            print('\n STOI Loss = ', bin_stoi_loss)
            loss += bin_stoi_loss

        if self.ild_weight > 0:
            ild_loss = ild_loss_db(target_stft_l.abs(), target_stft_r.abs(),
                                   output_stft_l.abs(), output_stft_r.abs())
            
            bin_ild_loss = self.ild_weight*ild_loss
            print('\n ILD Loss = ', bin_ild_loss)
            loss += bin_ild_loss

        if self.ipd_weight > 0:
            ipd_loss = ipd_loss_rads(target_stft_l, target_stft_r,
                                     output_stft_l, output_stft_r)
            bin_ipd_loss = self.ipd_weight*ipd_loss
            
            print('\n IPD Loss = ', bin_ipd_loss)
            loss += bin_ipd_loss
        
        return loss