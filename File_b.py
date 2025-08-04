import math
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP, Linear,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp


class TrendBlock(nn.Module):
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input,shape):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class FourierLayer(nn.Module):
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq
        
    def forward(self, x, shape):
        b, t_input, d = x.shape  # [128, 30, 64]
        t_target = shape  # 10

        # Calculate the original spectrum (based on input length t_input=30)
        x_freq = torch.fft.rfft(x, dim=1)  # [128, 16, 64] (因为 30//2 +1 =16)

        # Select the low-frequency section (skip the lowest frequency self.low_freq=1)
        if t_input % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]  # [128, 14, 64] (16-2=14)
            f = torch.fft.rfftfreq(t_input)[self.low_freq:-1]  # [14]
        else:
            x_freq = x_freq[:, self.low_freq:]  # [128, 15, 64] (16-1=15)
            f = torch.fft.rfftfreq(t_input)[self.low_freq:]  # [15]

        x_freq, index_tuple = self.topk_freq(x_freq)  # x_freq.shape = [128, k, 64]

        f = f.to(x_freq.device)  #
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))  # [128, k, 64]
        f = f[index_tuple]  # Filter by topk index
        f = rearrange(f, 'b f d -> b f () d')  # [128, k, 1, 64]

        return self.extrapolate(x_freq, f, t_target)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1, 
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()
        
        self.n_embd = n_embd
        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x

class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_shape,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        self.n_shape = n_shape
        self.trend = TrendBlock(n_channel, n_shape, n_embd, n_embd, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_shape, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat-2)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1,self.n_shape), self.seasonal(x2,self.n_shape)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season
    
class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_shape,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.n_shape = n_shape
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_shape=n_shape,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, _, _ = x.shape
        c = self.n_shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.d_model), device=x.device)

        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season

class GrangerNet(nn.Module):
    def __init__(self, lag=20, num_effects=26, hidden_size=32):
        super().__init__()
        self.lag = lag
        self.num_effects = num_effects

        # Define an independent MLP for each effect variable（component-wise）
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * lag-1, hidden_size), 
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            for _ in range(num_effects)
        ])

    def forward(self, x_exp, x_eff):
        B, L, _ = x_exp.shape
        _, _, F = x_eff.shape
        assert L == self.lag and F == self.num_effects

        x_exp_flat = x_exp.squeeze(-1)  # (B, L)

        outputs = []
        for i in range(F):
            x_eff_i = x_eff[:, :, i]       # (B, L)
            x_cat = torch.cat([x_eff_i, x_exp_flat], dim=1)  # (B, 2L)
            y = self.mlps[i](x_cat).squeeze(-1)              # (B,)
            outputs.append(y)

        preds = torch.stack(outputs, dim=1)  # (B, F)
        return preds
    
    def instancewise_scores(self, x_exp, x_eff):
        B, L, _ = x_exp.shape
        _, _, Fea = x_eff.shape
        assert Fea == self.num_effects

        x_exp_flat = x_exp.squeeze(-1)  # (B, L)
        scores = []

        for i in range(Fea):
            x_eff_i = x_eff[:, :-1, i]  # (B, L)
            x_cat = torch.cat([x_eff_i, x_exp_flat], dim=1)  # (B, 2L)

            # Forward propagation of the first two layers of MLP
            h = self.mlps[i][0](x_cat)         # Linear
            h = F.relu(h)                      # ReLU
            h = self.mlps[i][2](h)             # Linear
            # Take the absolute value of the activation intensity as the contribution score
            score = torch.abs(h.squeeze(-1))   # (B,)
            scores.append(score)

        return torch.stack(scores, dim=1)  # (B, F)

class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_shape,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.emb_cond = Conv_MLP(n_feat-2, n_embd, resid_pdrop=resid_pdrop)
        self.grangerLag = n_channel - n_shape if n_shape<n_channel else n_channel

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat-2, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_t = nn.Conv1d(n_embd, n_feat-2, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.dropout_s = nn.Dropout(p=resid_pdrop) 
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)
        self.dropout_m = nn.Dropout(p=resid_pdrop) 

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len-n_shape if n_shape<max_len else max_len)

        self.decoder = Decoder(n_channel,n_shape,n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=n_shape)

        self.GrangerNet = GrangerNet(lag=self.grangerLag, num_effects=n_feat-2)

    def forward(self, input, history, t, clip, padding_masks=None, return_res=False):
        rps_raw_data = history[:,:,0]
        rps_raw_grad = torch.diff(rps_raw_data, dim=0) 
        rps_raw_grad = torch.cat([torch.zeros(1, rps_raw_data.shape[1]).cuda(), rps_raw_grad], dim=0)
        
        rps_data = torch.cat([rps_raw_data.unsqueeze(-1),rps_raw_grad.unsqueeze(-1)],dim=-1) # 【batch_size, window-DeltaT,2】

        preds = self.GrangerNet(rps_data[:,:,0].unsqueeze(-1),history[:,:-1,1:])
        loss_causal = F.mse_loss(preds, history[:,-1,1:])
        causal_scores = self.GrangerNet.instancewise_scores(rps_data[:,:,0].unsqueeze(-1), history[:,:,1:])  # (B, F)
        mask = causal_scores > 0.1  
        causal_scores = causal_scores * mask

        # Causal weighting  
        cond_data = 3/10 * (causal_scores.unsqueeze(1)*rps_data[:,:,0].unsqueeze(-1) + (1-causal_scores.unsqueeze(1))*history[:,:,1:]) \
            + 7/10 * (causal_scores.unsqueeze(1)*rps_data[:,:,1].unsqueeze(-1) + (1-causal_scores.unsqueeze(1))*history[:,:,1:])
        
        # Integrate noise and conditional data as inputs
        input = torch.cat([torch.zeros(input.shape[0], input.shape[1], 2).cuda(),input], dim=-1)
        emb = self.emb(input) 
        inp_dec = self.pos_dec(emb)
        cond_feature = self.emb_cond(cond_data)
        inp_enc = self.pos_enc(cond_feature)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)
        
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
        
        return self.dropout_m(self.combine_t((trend+output).transpose(1, 2)).transpose(1, 2)), \
            self.dropout_s(self.combine_s(season.transpose(1, 2)).transpose(1, 2)), loss_causal

    def _ssf_input(self, x: torch.Tensor):
        batch_size = x.size()[:-1]
        x = x.reshape(self.scale2.shape[0], -1, x.shape[-1])
        return (x * self.scale2).view(*batch_size, x.shape[-1])
    