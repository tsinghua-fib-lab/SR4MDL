import os
import time
import json
import torch
import psutil
import logging
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import PercentFormatter
from .utils import MLP, noam_lambda, Timer, NamedTimer, AttentionPooling, CLIPLoss
from ..utils import AttrDict, R2_score, RMSE_score, AUC_score

import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd")


class MDLformer(nn.Module):
    def __init__(self, args:AttrDict, token_list:List[str]):
        super(MDLformer, self).__init__()
        self.args = args
        self.idx2token = token_list
        self.token2idx = {token: idx for idx, token in enumerate(token_list)}
        self.padding_idx = self.token2idx['PAD']

        self.embedding = nn.Embedding(len(token_list), args.d_input, padding_idx=self.padding_idx)
        self.linear = MLP((args.max_var+1) * 3 * args.d_input,
                          (args.max_var+1) * 3 * args.d_input,
                          args.d_model, dropout=args.dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.d_model//64,
                dim_feedforward=args.d_model*4,
                dropout=args.dropout,
                batch_first=True),
            num_layers=args.n_TE_layers
        )
        self.pooling = AttentionPooling(args.d_model)
        self.bottleneck = nn.Linear(args.d_model, args.d_output)
        self.output = MLP(args.d_output, args.d_output, 1, out_act=nn.Identity, dropout=args.dropout, dropout_output=False)

        self.to(args.device)

    def __repr__(self):
        return str(self)

    def __str__(self):
        infos = []
        for name, param in self.named_parameters():
            # Name, #Params, Trainable, Device
            infos.append((name, param.numel(), "(TRAIN)" if param.requires_grad else "(FREEZE)", str(param.device)))
        infos = sorted(infos, key=lambda x: x[1], reverse=True)
        return f'({self.__class__.__name__}) Total Trainable Params: {sum(size for _, size, trainable, _ in infos if trainable == "(TRAIN)"):,}'
        # rows = []
        # rows.append(f'{"Name":60} {"#Params":12} {"Trainable":10} {"Device"}')
        # rows.extend([f'{name:60} {size:12,} {trainable:10} {device}' for name, size, trainable, device in infos])
        # return '\n'.join(rows)

    def to(self, device):
        self.device = device
        return super().to(device)

    def preprocess(self, data: torch.LongTensor|List[np.ndarray]):
        """
        data: List of (N_i, D_max+1, 3) array or (B, N_max, D_max+1, 3) tensor
            - When {data} is a tensor, it is already the padded index tensor
            - When {data} is a list of array, it is a list of (N_i, D_max+1, 3) array of token
            - When self.args.uniform_sample_number == True, N_i == N_max for all N_i
        """
        device = self.embedding.weight.device
        if isinstance(data, torch.Tensor):
            """ {data} is already the padded index tensor """
            data = data.to(device)
        elif self.args.uniform_sample_number:
            """ {data} is a list of (N, D_max+1, 3) ndarray of token, with no need of padding """
            data = np.stack(data, axis=0)
            data = (np.vectorize(self.token2idx.get))(data)
            data = torch.LongTensor(data)
            data = data.to(device) # (B, N_max, D_max+1, 3)
        else:
            """ {data} is a list of (Ni, D_max+1, 3) ndarray of token """
            vectorize = np.vectorize(self.token2idx.get)
            for i, d in enumerate(data):
                data[i] = torch.LongTensor(vectorize(d)).to(device)
            data = pad_sequence(data, batch_first=True, padding_value=self.padding_idx)  # (B, N_max, D_max+1, 3)
        return data

    def forward(self, data: torch.LongTensor|List[np.ndarray]):
        """
        data: List of (N_i, D_max+1, 3) ndarray or (B, N_max, D_max+1, 3) tensor
        """
        # PREPROCESS
        x = self.preprocess(data)  # (B, N_max, D_max+1, 3)
        src_key_padding_mask = (x[..., -1, 0] == self.padding_idx)  # (B, N_max)

        # EMBEDDING
        x = self.embedding(x)  # (B, N_max, D_max+1, 3, d_input)

        # SQUEEZE & EXCITATION
        x = self.linear(x.flatten(-3, -1))  # (B, N_max, model)

        # PROCESS
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, N_max, d_model)

        # POOLING
        x = self.pooling(x, src_key_padding_mask)  # (B, d_model)
        x = self.bottleneck(x)  # (B, d_output)
        saved_bottleneck = x

        # MLP
        x = self.output(x).squeeze(-1)  # (B,)
        return x, saved_bottleneck

    def predict(self, data:torch.tensor):
        with torch.no_grad():
            # PREPROCESS
            x = self.preprocess(data)  # (B, N_max, D_max+1, 3)
            src_key_padding_mask = (x[..., -1, 0] == self.padding_idx)  # (B, N_max)

            # EMBEDDING
            x = self.embedding(x)  # (B, N_max, D_max+1, 3, d_model)

            # SQUEEZE & EXCITATION
            x = self.linear(x.flatten(-3, -1))  # (B, N_max, d_input)

            # PROCESS
            with torch.amp.autocast('cuda'):
                x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, N_max, d_model)

            # POOLING
            x = self.pooling(x, src_key_padding_mask) # (B, d_model)
            x = self.bottleneck(x)  # (B, d_output)

            # MLP
            x = self.output(x).squeeze(-1)  # (B,)
            return x

    def load(self, state_dict, token_list, strict=True):
        if token_list != self.idx2token:
            if strict: raise ValueError(f"Token list mismatch: {set(token_list).symmetric_difference(self.idx2token)}")
            mapping = {idx: self.token2idx[token] for idx, token in enumerate(token_list) if token in self.token2idx}
            tmp = self.state_dict()['embedding.weight'].clone()
            for old_idx, new_idx in mapping.items():
                tmp.data[new_idx] = state_dict['embedding.weight'].data[old_idx]
            state_dict['embedding.weight'] = tmp
        self.load_state_dict(state_dict, strict=strict)


class FormulaEncoder(MDLformer):
    def __init__(self, args:AttrDict, token_list:List[str]):
        super(FormulaEncoder, self).__init__(args, token_list)
        self.embedding = nn.Embedding(len(token_list), args.d_model, padding_idx=self.padding_idx).to(args.device)
        delattr(self, 'linear')
        delattr(self, 'output')

    def preprocess(self, data:torch.LongTensor|List[np.ndarray]):
        """
        data: List of (N_i,) ndarray or (B, N_max) tensor
            - When {data} is a tensor, it is already the padded index tensor
            - When {data} is a list of array, it is a list of (N_i,) array of token
        """
        device = self.embedding.weight.device
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        else:
            vectorize = np.vectorize(self.token2idx.get)
            for i, p in enumerate(data):
                data[i] = torch.LongTensor(vectorize(p)).to(device)
            data = pad_sequence(data, batch_first=True, padding_value=self.padding_idx) # (B, N_max)
        return data

    def forward(self, data:torch.LongTensor|List[np.ndarray]):
        """
        data: List of (N_i,) ndarray or (B, N_max) tensor
        """
        # TOKEN to INDEX
        x = self.preprocess(data)  # (B, N_max)
        src_key_padding_mask = (x == self.padding_idx)  # (B, N_max)

        # EMBEDDING
        x = self.embedding(x)  # (B, N_max, d_model)

        # PROCESS
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, N_max, d_model)

        # POOLING
        x = self.pooling(x, src_key_padding_mask)  # (B, d_model)
        x = self.bottleneck(x)  # (B, d_output)
        return x


class Trainer:
    def __init__(self, args:AttrDict, mdlformer:MDLformer, eq_encoder:FormulaEncoder):
        self.args = args
        self.mdlformer = mdlformer
        self.eq_encoder = eq_encoder
        self.optimizer = torch.optim.Adam([
            {'params': self.mdlformer.parameters()},
            {'params': self.eq_encoder.parameters()}
        ], lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=noam_lambda(args.d_model, args.warmup_steps)
        )
        self.criterion1 = CLIPLoss(clip_temperature=1.0)
        self.criterion2 = torch.nn.MSELoss()

        self.n_step = 0
        self.records = []
        self.logger = logging.getLogger('my.trainer')
        self.timer = Timer()  # 测量速度
        self.named_timer = NamedTimer()  # 测量时间分配 (load data, forward, backward, evaluate, save & plot)

    def step(self,
             data:torch.LongTensor|List[np.ndarray],
             prefix:torch.LongTensor|List[np.ndarray],
             used_vars:torch.LongTensor|List[np.ndarray],
             mdl:torch.LongTensor|List[int],
             detail=True):
        """
        - data: List of (N_i, D_max+1, 3) ndarray or (B, N_max, D_max+1, 3) tensor
        - prefix: List of (N_i,) ndarray or (B, N_max) tensor
        - used_vars: List of (N_i,) ndarray or (B, N_max) tensor
        - mdl: List of int or tensor
        """
        self.n_step += 1
        self.timer.add(len(data))
        self.named_timer.add('load data')
        record = {'step': self.n_step}  # for save
        log = {'Step': f'{record["step"]:03d}'}  # for print

        if isinstance(mdl, list): mdl = torch.FloatTensor(mdl)
        mdl = mdl.float().to(self.args.device)
        mdl.clip_(0, self.args.max_len-1)

        if isinstance(used_vars, list): used_vars = torch.FloatTensor(used_vars)
        used_vars = used_vars.float().to(self.args.device)

        if self.args.AMP:
            # raise NotImplementedError
            with autocast('cuda'):
                pred, xy_embedding = self.mdlformer(data)
                eq_embedding = self.eq_encoder(prefix)
                clip_loss = self.criterion1(xy_embedding.float(), eq_embedding.float())
                pred_loss = self.criterion2(pred.float(), mdl.float())
                pred_loss_scale = self.args.pred_loss_weight
                clip_loss_scale = self.args.clip_loss_weight
                if self.args.normalize_loss:
                    pred_loss_scale /= (pred_loss.item() + 1e-6)
                    clip_loss_scale /= (clip_loss.item() + 1e-6)
                loss = 0
                if pred_loss_scale != 0: loss = loss + pred_loss_scale * pred_loss
                if clip_loss_scale != 0: loss = loss + clip_loss_scale * clip_loss
            self.named_timer.add('forward')
            self.optimizer.zero_grad()
            scaler = GradScaler('cuda')
            scaler.scale(loss).backward()
            if self.args.grad_clip:
                scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.mdlformer.parameters(), max_norm=1.0)
                clip_grad_norm_(self.eq_encoder.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            self.named_timer.add('backward')
        else:
            pred, xy_embedding = self.mdlformer(data)
            eq_embedding = self.eq_encoder(prefix)
            clip_loss = self.criterion1(xy_embedding, eq_embedding)
            pred_loss = self.criterion2(pred, mdl)
            pred_loss_scale = self.args.pred_loss_weight
            clip_loss_scale = self.args.clip_loss_weight
            if self.args.normalize_loss:
                pred_loss_scale /= (pred_loss.item() + 1e-6)
                clip_loss_scale /= (clip_loss.item() + 1e-6)
            loss = 0
            if pred_loss_scale != 0: loss = loss + pred_loss_scale * pred_loss
            if clip_loss_scale != 0: loss = loss + clip_loss_scale * clip_loss
            self.named_timer.add('forward')
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip:
                clip_grad_norm_(self.mdlformer.parameters(), max_norm=1.0)
                clip_grad_norm_(self.eq_encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.named_timer.add('backward')
        record['lr'] = self.optimizer.param_groups[0]['lr']
        record['batch_size'] = len(data)
        n_gpus = torch.cuda.device_count()
        record['mem_alloc'] = sum(torch.cuda.memory_allocated(idx) for idx in range(n_gpus)) / 1024 ** 2
        record['mem_reserv'] = sum(torch.cuda.memory_reserved(idx) for idx in range(n_gpus)) / 1024 ** 2
        record['mem_max_alloc'] = sum(torch.cuda.max_memory_allocated(idx) for idx in range(n_gpus)) / 1024 ** 2
        record['mrm_max_reserv'] = sum(torch.cuda.max_memory_reserved(idx) for idx in range(n_gpus)) / 1024 ** 2
        record['main_memory'] = psutil.Process().memory_info().rss / 1024 ** 2
        log['LR'] = f'{record["lr"]:.02e}'
        log['Mem'] = f'{record["mem_alloc"]:.0f}@{record["mem_reserv"]:.0f}MiB (peak={record["mem_max_alloc"]:.0f}@{record["mrm_max_reserv"]:.0f}MiB)'
        log['MainMem'] = f'{record["main_memory"]:,.0f}MiB'
        if detail:
            pred = pred.cpu().detach().tolist()
            true = mdl.cpu().detach().tolist()
            record['pred'] = pred
            record['true'] = true
            record['RMSE'] = RMSE_score(true, pred)
            record['AUC'] = AUC_score(true, pred)
            record['R2'] = R2_score(true, pred)
            record['loss'] = loss.item()
            record['clip_loss'] = clip_loss.item()
            record['pred_loss'] = pred_loss.item()
            record['speed'] = self.timer.pop()
            record['time'] = self.named_timer.pop()

            log['RMSE'] = f'{record["RMSE"]:.2f}'
            log['AUC'] = f'{record["AUC"]:.2%}'
            log['R2'] = f'{record["R2"]:.3f}'
            log['Loss'] = f'{record["loss"]:.4f}'
            log['ClipLoss'] = f'{record["clip_loss"]:.4f}'
            log['PredLoss'] = f'{record["pred_loss"]:.4f}'
            log['Speed'] = f'{record["speed"]:.1f}eq/s'
            tot = sum(record['time'].values())
            log['Time'] = f'{tot*1000:.0f}ms (' + ','.join(f'{n}={t/tot:.0%}' for n, t in record['time'].items()) + ')'
        self.named_timer.add('evaluate')
        self.records.append(record)
        with open(os.path.join(self.args.save_dir, 'records.json'), 'a') as f:
            f.write(json.dumps(record) + '\n')

        if not self.n_step % 500: self.plot('plot.png')
        if not self.n_step % 1000: self.save('checkpoint.pth')
        if all(i == '0' for i in str(self.n_step)[1:]):
            if self.n_step >= 10: self.plot(f'plots/plot_{self.n_step}.png')
            if self.n_step >= 100: self.save(f'checkpoints/checkpoint_{self.n_step}.pth', model_only=True)
        self.named_timer.add('save')
        return log

    def plot(self, path, abs_path=False):
        from ..utils.plot import get_fig
        fi, fig, gs = get_fig(4, 3, FW=14, A_ratio=1.0, gridspec=True)
        axes = {
            'Loss': fig.add_subplot(gs[0, 0]),
            'CLIPLoss': fig.add_subplot(gs[0, 1]),
            'PredLoss': fig.add_subplot(gs[0, 2]),
            'RMSE': fig.add_subplot(gs[1, 0]),
            'AUC': fig.add_subplot(gs[1, 1]),
            'R2': fig.add_subplot(gs[1, 2]),
            'R2-demo': fig.add_subplot(gs[2, 0]),
            'LR': fig.add_subplot(gs[2, 1]),
            'BatchSize': fig.add_subplot(gs[2, 2]),
            'Mem': fig.add_subplot(gs[3, 0]),
            'Speed': fig.add_subplot(gs[3, 1]),
            'Time': fig.add_subplot(gs[3, 2]),
        }
        cmap = LinearSegmentedColormap.from_list('g2r', ['#00b894', '#fdcb6e', '#e17055', '#d63031'])
        norm = Normalize(vmin=0, vmax=self.args.max_len)
        for ax in axes.values():
            ax.grid('on', linestyle='-.', alpha=0.5, color='#dfe6e9', zorder=0)
            ax.tick_params(axis='both', direction='in', length=2, pad=2)

        def get_xy(key):
            xy = [(r['step'], r[key]) for r in self.records if key in r]
            x, y = list(zip(*xy)) if xy else ([], [])
            return np.array(x), np.array(y)

        def get_smooth_xy(key, N=20):
            x, y = get_xy(key)
            if len(x) < N: return x, y
            xy_list = np.array_split(np.stack([x, y], axis=0), N, axis=1)
            x2, y2 = list(zip(*[xy.mean(axis=1) for xy in xy_list]))
            return np.array(x2), np.array(y2)

        x, y = get_xy('loss')
        axes['Loss'].plot(x, y, color='#0984e3', alpha=0.3, zorder=3, label='Total Loss')
        axes['Loss'].plot(*get_smooth_xy('loss'), '-o', markersize=3, color='#0984e3', zorder=6)
        axes['Loss'].set_title(f'Loss ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        x, y = get_xy('clip_loss')
        axes['CLIPLoss'].plot(x, y, color='#0984e3', alpha=0.3, zorder=1, label='Clip Loss')
        axes['CLIPLoss'].plot(*get_smooth_xy('clip_loss'), '-o', markersize=3, color='#0984e3', zorder=5)
        axes['CLIPLoss'].set_title(f'CLIP Loss ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        x, y = get_xy('pred_loss')
        axes['PredLoss'].plot(x, y, color='#0984e3', alpha=0.3, zorder=2, label='Pred Loss')
        axes['PredLoss'].plot(*get_smooth_xy('pred_loss'), '-o', markersize=3, color='#0984e3', zorder=4)
        axes['PredLoss'].set_title(f'Pred Loss ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        M = self.args.max_len

        x, y = get_xy('RMSE')
        axes['RMSE'].plot(x, y, color='#0984e3', alpha=0.3, zorder=M+1)
        axes['RMSE'].plot(*get_smooth_xy('RMSE'), '-o', markersize=3, color='#0984e3', zorder=M+1+M)
        axes['RMSE'].set_title(f'Root Mean Squared Error ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        x, y = get_xy('AUC')
        axes['AUC'].plot(x, y, color='#0984e3', alpha=0.3, zorder=M+1)
        axes['AUC'].plot(*get_smooth_xy('AUC'), '-o', markersize=3, color='#0984e3', zorder=M+1+M)
        axes['AUC'].set_title(f'AUC-ROC ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        x, y = get_xy('R2')
        axes['R2'].plot(x, y, color='#0984e3', alpha=0.3)
        axes['R2'].plot(*get_smooth_xy('R2'), '-o', markersize=3, color='#0984e3')
        axes['R2'].set_ylim(-0.1, 1.1)
        axes['R2'].set_title(f'R2 ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        x, y = [], []
        for rec in self.records[-100:]:
            if 'true' in rec:
                x.extend(rec['true'])
                y.extend(rec['pred'])
        axes['R2-demo'].plot(x, y, 'o', markersize=2, color='#0984e3', alpha=0.3, markeredgecolor='none')
        axes['R2-demo'].plot([0, M], [0, M], '--', color='#d63031')
        axes['R2-demo'].set_title('Pred vs True')
        axes['R2-demo'].set_ylabel('Pred')
        axes['R2-demo'].set_xlabel('True')

        axes['LR'].plot(*get_xy('lr'), color='#0984e3')
        axes['LR'].set_title('LR')

        axes['BatchSize'].plot(*get_xy('batch_size'), color='#0984e3')
        axes['BatchSize'].set_title('Batch Size')

        axes['Mem'].plot(*get_xy('mem_alloc'), color='#0984e3', label='CUDA Allocated')
        axes['Mem'].plot(*get_xy('mem_reserv'), color='#e17055', label='CUDA Reserved')
        axes['Mem'].plot(*get_xy('mem_max_alloc'), color='#00b894', label='CUDA Max Allocated')
        axes['Mem'].plot(*get_xy('mrm_max_reserv'), color='#d63031', label='CUDA Max Reserved')
        axes['Mem'].plot(*get_xy('main_memory'), color='#fdcb6e', label='Main Memory')
        axes['Mem'].set_title('Memory (MiB)')
        axes['Mem'].legend(ncol=1, fontsize=5, handlelength=0.5, handletextpad=0.5)

        x, y = get_xy('speed')
        axes['Speed'].plot(x, y, color='#0984e3', alpha=0.3)
        axes['Speed'].plot(*get_smooth_xy('speed'), '-o', markersize=3, color='#0984e3')
        axes['Speed'].set_title(f'Speed (Equation / s) ({y[-100:].mean():.2f}±{y[-100:].std():.2f})')

        for rec in self.records:
            if 'time' in rec: break
        if 'time' in rec:
            x = []
            p = {k: [] for k in rec['time']}
            for rec in self.records:
                if 'time' not in rec: continue
                for k, v in rec['time'].items(): p[k].append(v)
                x.append(rec['step'])
            k = list(p.keys())
            p = np.array(list(p.values())).cumsum(axis=0)
            p = np.concatenate([np.zeros_like(p[(0,), :]), p], axis=0).cumsum(axis=0)
            for i in range(5):
                axes['Time'].fill_between(x, p[i], p[i+1], color=cmap(i/5), alpha=0.3, label=k[i], ec='none')
            axes['Time'].plot(x, p[-1], color='#0984e3')
            axes['Time'].legend(ncol=1, fontsize=0.7*fi.fontsize)
            axes['Time'].set_title('Time (s)')
        axes['Time2'] = axes['Time'].twinx()
        axes['Time2'].plot(x, np.cumsum(p[-1]), color='#e17055')
        def seconds_to_hms(x, pos):
            d = int(x // 86400)
            h = int((x % 86400) // 3600)
            m = int((x % 3600) // 60)
            s = int(x % 60)
            return (f'{d}d ' if d > 0 else '') + f'{h:02}:{m:02}:{s:02}'
        axes['Time2'].yaxis.set_major_formatter(plt.FuncFormatter(seconds_to_hms))
        axes['Time2'].tick_params(axis='both', direction='in', length=2, pad=2)

        fig.tight_layout()
        if not abs_path: path = os.path.join(self.args.save_dir, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)

        # 避免内存泄漏
        plt.close(fig)
        for ax in axes.values(): ax.clear()
        fig.clear()
        del fig, axes

    def save(self, path, abs_path=False, model_only=False):
        if not abs_path: path = os.path.join(self.args.save_dir, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(self.mdlformer, nn.DataParallel):
            state_dict = {
                'xy_encoder': self.mdlformer.module.state_dict(),
                'xy_token_list': self.mdlformer.module.idx2token,
                'eq_encoder': self.eq_encoder.module.state_dict(),
                'eq_token_list': self.eq_encoder.module.idx2token,
            }
        else:
            state_dict = {
                'xy_encoder': self.mdlformer.state_dict(),
                'xy_token_list': self.mdlformer.idx2token,
                'eq_encoder': self.eq_encoder.state_dict(),
                'eq_token_list': self.eq_encoder.idx2token,
            }
        if not model_only:
            state_dict.update({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            })
        state_dict.update({
            'args': self.args,
            'step': self.n_step
        })
        torch.save(state_dict, path)
        self.logger.note(f'Saved checkpoint to {path}')

    def load(self, path, abs_path=False, model_only=False):
        if not abs_path: path = os.path.join(self.args.save_dir, path)
        state_dict = torch.load(path, map_location=self.args.device, weights_only=False)

        if model_only and 'optimizer' in state_dict:
            self.logger.warning('Optimizer/scheduler in checkpoint but not loaded')
        if not model_only and 'optimizer' not in state_dict:
            self.logger.warning('No optimizer/scheduler in checkpoint')
            model_only = True

        state_dict['xy_encoder'] = {k.removeprefix('module.'): v for k, v in state_dict['xy_encoder'].items()}
        state_dict['eq_encoder'] = {k.removeprefix('module.'): v for k, v in state_dict['eq_encoder'].items()}

        if isinstance(self.mdlformer, nn.DataParallel):
            self.mdlformer.module.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=not model_only)
            self.eq_encoder.module.load(state_dict['eq_encoder'], state_dict['eq_token_list'], strict=not model_only)
        else:
            self.mdlformer.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=not model_only)
            self.eq_encoder.load(state_dict['eq_encoder'], state_dict['eq_token_list'], strict=not model_only)
        if not model_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.n_step = state_dict['step']
        self.logger.info(f'Loaded checkpoint from {path}')
