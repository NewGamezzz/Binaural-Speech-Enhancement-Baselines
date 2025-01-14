import torch
import torch.nn as nn
from .feature_extractors import IStft, Stft

from .utils.complexPyTorch import complexLayers as torch_complex
from .utils.show import show_params, show_model

from .utils.apply_mask import apply_mask


class DCNN(nn.Module):
    def __init__(
        self,
        rnn_layers=2,
        rnn_units=128,
        win_len=400,
        win_inc=100,
        fft_len=512,
        win_type="hann",
        masking_mode="E",
        use_clstm=False,
        kernel_size=5,
        kernel_num=[16, 32, 64, 128, 256, 256],
        # kernel_num = [ 8, 16, 32, 64, 128, 128],
        bidirectional=False,
        embed_dim=1024,
        num_heads=32,
        **kwargs
    ):
        """
        rnn_layers: the number of lstm layers in the crn,
        rnn_units: for clstm, rnn_units = real+imag
        """

        super().__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc  # TODO: Rename to hop_size
        self.fft_len = fft_len

        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        # self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        # self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        self.stft = Stft(self.fft_len, self.win_inc, self.win_len)
        self.istft = IStft(self.fft_len, self.win_inc, self.win_len)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num) + 1))
        self.mattn = MultiAttnBlock(
            input_size=1024,
            hidden_size=self.rnn_units,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )

        self.encoder = Encoder(self.kernel_num, kernel_size)
        # self._create_rnn(rnn_layers)
        # self.attn = FAL(in_channels=1, out_channels=96, f_length=256)

        # self.rnn = RnnBlock(
        #     # if idx == 0 else self.rnn_units,
        #     input_size=hidden_dim * self.kernel_num[-1],
        #     hidden_size=self.rnn_units,
        #     bidirectional=bidirectional,
        #     num_layers=rnn_layers)

        self.decoder = Decoder(self.kernel_num, self.kernel_size)

        # show_model(self)
        # show_params(self)
        # self._flatten_parameters()

    def forward(self, inputs):

        # 0. Extract STFT
        x = cspecs = self.stft(inputs)
        x = x.unsqueeze(1)  # Add a dummy channel

        # x=self.attn(x)

        encoder_out = self.encoder(x)
        x = encoder_out[-1]

        # 2. Apply RNN
        x = self.mattn(x)

        # 3. Apply decoder
        x = self.decoder(x, encoder_out)

        # 4. Apply mask
        out_spec = apply_mask(x[:, 0], cspecs, self.masking_mode)

        # 5. Invert STFT
        out_wav = self.istft(out_spec)
        # out_wav = torch.squeeze(out_wav, 1)
        # out_wav = torch.clamp_(out_wav, -1, 1)
        # breakpoint()
        return out_wav  # out_spec, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if "bias" in name:
                biases += [param]
            else:
                weights += [param]
        params = [
            {
                "params": weights,
                "weight_decay": weight_decay,
            },
            {
                "params": biases,
                "weight_decay": 0.0,
            },
        ]
        return params


class BinauralAttentionDCNN(DCNN):
    def forward(self, inputs):
        batch_size, binaural_channels, time_bins = inputs.shape
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)

        # breakpoint()

        # encoder_out_l = self.encoder(attention_enc[:, 0, :, :].unsqueeze(1))
        # encoder_out_r = self.encoder(attention_enc[:, 1, :, :].unsqueeze(1))

        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        # breakpoint()

        # encoder_out = torch.cat((encoder_out_l[-1], encoder_out_r[-1]), dim=1)
        # attention_enc = self.attention_enc(encoder_out)
        # breakpoint()
        # attention_enc = encoder_out_l[-1]*encoder_out_r[-1].conj()
        # attention_enc = self.attention(cspecs)
        # _, attn_len, _, _ = attention_enc.shape
        # encoder_attn_l = attention_enc[:, :attn_len//2, :, :]
        # encoder_attn_r = attention_enc[:, attn_len//2:, :, :]
        # breakpoint()
        # 2. Apply RNN
        # x_l_rnn = self.rnn(encoder_out_l[-1])
        # x_r_rnn = self.rnn(encoder_out_r[-1])
        # breakpoint()
        attention_in = torch.cat((encoder_out_l[-1], encoder_out_r[-1]), dim=1)

        #
        # breakpoint()
        # x_l_mattn = self.mattn(encoder_out_l[-1])
        # x_r_mattn = self.mattn(encoder_out_r[-1])
        # breakpoint()

        x_attn = self.mattn(attention_in)
        # rnn_out = torch.cat((x_l_rnn, x_r_rnn), dim=1)
        # breakpoint()
        # attention_dec = self.attention_enc(rnn_out)

        # _, dec_attn_len, _, _ = attention_dec.shape
        # decoder_attn_l = attention_dec[:, :dec_attn_len//2, :, :]
        # decoder_attn_r = attention_dec[:, dec_attn_len//2:, :, :]
        x_l_mattn = x_attn[:, :128, :, :]
        x_r_mattn = x_attn[:, 128:, :, :]
        # x_l_mattn = x_attn[:,:64,:,:]
        # x_r_mattn = x_attn[:,64:,:,:]
        # 3. Apply decoder
        # x_l = self.decoder(x_l_rnn, encoder_out_l)
        # x_r = self.decoder(x_r_rnn, encoder_out_r)
        x_l = self.decoder(x_l_mattn, encoder_out_l)
        x_r = self.decoder(x_r_mattn, encoder_out_r)

        # 4. Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

        # 5. Invert STFT
        out_wav_l = self.istft(out_spec_l, length=time_bins)
        # breakpoint()
        # out_wav_l = torch.squeeze(out_wav_l, 1)
        # out_wav_l = torch.clamp_(out_wav_l, -1, 1)

        out_wav_r = self.istft(out_spec_r, length=time_bins)
        # out_wav_r = torch.squeeze(out_wav_r, 1)
        # out_wav_r = torch.clamp_(out_wav_r, -1, 1)

        # breakpoint()

        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)

        return out_wav


class Encoder(nn.Module):
    def __init__(self, kernel_num, kernel_size):
        super().__init__()

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.model.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    torch_complex.ComplexConv2d(
                        self.kernel_num[idx] // 2,
                        self.kernel_num[idx + 1] // 2,
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                    ),
                    torch_complex.NaiveComplexBatchNorm2d(self.kernel_num[idx + 1] // 2),
                    torch_complex.ComplexPReLU(),
                )
            )

    def forward(self, x):
        output = []
        # 1. Apply encoder
        for idx, layer in enumerate(self.model):
            x = layer(x)
            # x = x[..., :-1] # Experimental
            output.append(x)

        return output


class Decoder(nn.Module):
    def __init__(self, kernel_num, kernel_size):
        super().__init__()

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        self.model = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            block = [
                torch_complex.ComplexConvTranspose2d(
                    self.kernel_num[idx],  # * 2,
                    self.kernel_num[idx - 1] // 2,
                    kernel_size=(self.kernel_size, 2),
                    stride=(2, 1),
                    padding=(2, 1),
                    output_padding=(1, 0),
                ),
            ]

            if idx != 1:
                block.append(torch_complex.NaiveComplexBatchNorm2d(self.kernel_num[idx - 1] // 2))
                block.append(torch_complex.ComplexPReLU())
            self.model.append(nn.Sequential(*block))

    def forward(self, x, encoder_out):
        for idx in range(len(self.model)):
            # x = complex_cat([x, encoder_out[-1 - idx]], 1)
            x = torch.cat([x, encoder_out[-1 - idx]], 1)
            x = self.model[idx](x)
            # x = x[..., 1:]

        return x


# class RnnBlock(nn.Module):
#     def __init__(self, input_size, hidden_size, bidirectional, num_layers) -> None:
#         super().__init__()

#         self.rnn = torch_complex.ComplexLSTM(
#             input_size=input_size,  # if idx == 0 else self.rnn_units,
#             hidden_size=hidden_size,
#             bidirectional=bidirectional,
#             num_layers=num_layers,
#             batch_first=True
#         )

#         self.transform = nn.Linear(
#             hidden_size,
#             input_size,
#             dtype=torch.complex64
#         )

#     def forward(self, x):
#         batch_size, channels, freqs, time_bins = x.shape
#         x = x.flatten(start_dim=1, end_dim=2)
#         x = x.transpose(1, 2)  # (batch_size, time_bins, rnn_channels)
#         # breakpoint()
#         x = self.rnn(x)[0]
#         x = self.transform(x)
#         # breakpoint()
#         x = x.unflatten(-1, (channels, freqs))
#         x = x.movedim(1, -1)

#         return x


class MultiAttnBlock(nn.Module):
    def __init__(self, input_size, hidden_size, embed_dim=128, num_heads=8, batch_first=True):
        super().__init__()

        self.mattn = torch_complex.ComplexMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first
        )

        self.transform = nn.Linear(
            in_features=input_size, out_features=input_size, dtype=torch.complex64
        )

    def forward(self, x):

        batch_size, channels, freqs, time_bins = x.shape
        x = x.flatten(start_dim=1, end_dim=2)
        x = x.transpose(1, 2)
        # breakpoint()
        x = self.mattn(x)
        # breakpoint()
        x = self.transform(x)
        x = x.unflatten(-1, (channels, freqs))
        x = x.movedim(1, -1)

        return x
