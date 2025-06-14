import os
import torch
import random
import logging
import itertools
import numpy as np
import sympy as sp
import nd2py as nd
import torch.utils.data as D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import islice
from typing import List, Tuple
from nd2py.utils import AttrDict, get_fig
from torch.nn.utils.rnn import pad_sequence
from .data_generator import GMMGenerator, SubeqGenerator
from .eqtree_generator import SNIPGenerator, SNIPGenerator3, SimpleGenerator
from ..env import Tokenizer, sympy2eqtree, str2sympy, decompose

logger = logging.getLogger(__name__)


class Num2EqDataset(D.Dataset):
    """
    - f ~ F 
        1. 生成 N 个变量和 M 个二元运算符的树 
        2. 向树中插入一元运算符 
        3. 在可能的地方插入数值常数 
        4. (optional) 化简
    - X ~ GMM(D)
        1. GMM
        2. 归一化到 mu=0, std=1
    - y = f(X)
        1. 移除 NaN
        2. (optional) 归一化到 1
    return [X, y], f
    """
    def __init__(self, args:AttrDict, beyond_token=False):
        super().__init__()
        self.args = args
        self.tokenizer = Tokenizer(-100, 100, 4, args.max_var) # -9.999e100 ~ -0.001e-100 ~ 0.0000 ~ 0.001e-100 ~ 9.999e100
        self.data_generator = GMMGenerator(max_value=self.tokenizer.MAX_VALUE)
        self.eqtree_generator = SNIPGenerator(**args)
        self.beoynd_token = beyond_token

    def __len__(self):
        return None

    def __getitem__(self, idx):
        while True:
            eqtree = self.eqtree_generator.generate_eqtree()
            if self.args.simplify:
                sympy_expr = str2sympy(str(eqtree))
                sympy_expr = sympy_expr.subs({sym: sp.Symbol(sym.name, real=True) for sym in sympy_expr.free_symbols})
                # sympy_expr = sp.simplify(sympy_expr)
                try:
                    _eqtree = sympy2eqtree(sympy_expr)
                except:
                    continue
                if len(_eqtree) <= len(eqtree): eqtree = _eqtree

            if not any(isinstance(x, nd.Variable) for x in eqtree.iter_preorder()): continue
            if len(eqtree) > self.args.max_len: continue

            N = self.args.uniform_sample_number or np.random.randint(100, 500)
            X, Y = self.data_generator.generate_data(N, eqtree)
            if X is None: continue
            if self.args.normalize_y: Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-6)

            data = np.pad(X, ((0, 0), (0, self.args.max_var - X.shape[1])), constant_values=0)
            data = np.concatenate([data, Y[:, None]], axis=1)
            length = len(eqtree)
            if self.args.save_equations:
                with open(os.path.join(self.args.save_dir, 'equations.txt'), 'a') as f:
                    f.write(f'{eqtree}\n')
                self.args.save_equations -= 1

            used_vars = [1] * X.shape[1] + [0] * (self.args.max_var - X.shape[1])

            if not self.beoynd_token:
                return self.tokenizer.float2token(data), self.tokenizer.eqtree2token(eqtree), used_vars, length
            return self.tokenizer.float2index(data), self.tokenizer.eqtree2index(eqtree), used_vars, length

    def collate_fn(self, batch:List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]):
        data, prefix, used_vars, length = zip(*batch)
        assert not any([np.isnan(d).any() for d in data]), 'FOR DEBUG'
        if not self.beoynd_token:
            return list(data), list(prefix), list(used_vars), list(length)
        data = list(data)
        for i, d in enumerate(data):
            data[i] = torch.LongTensor(d)
        data = pad_sequence(data, batch_first=True, padding_value=0) # (B, N_max, D_max+1, 3)
        prefix = list(prefix)
        for i, p in enumerate(prefix):
            prefix[i] = torch.LongTensor(p)
        prefix = pad_sequence(prefix, batch_first=True, padding_value=0) # (B, N_max)
        used_vars = torch.tensor(used_vars)
        return data, prefix, used_vars, torch.LongTensor(length)

    def get_dataloader(self, **kwargs):
        return D.DataLoader(self, collate_fn=self.collate_fn, sampler=InfiniteSampler(), **kwargs)

    def get_token_list(self, all=False):
        token_list = self.tokenizer.get_token_list(all=all)
        token_list.insert(0, 'PAD')
        return token_list


class Num2EqDatasetHard(Num2EqDataset):
    """
    生成的数据分布 X 是非常奇怪的，因此称为 Hard
    - f ~ F (同 default)
        1. 生成 N 个变量和 M 个二元运算符的树 
        2. 向树中插入一元运算符 
        3. 在可能的地方插入数值常数 
        4. (optional) 化简
    - X = phi(Z), Z ~ GMM(K), phi_{1..D} ~ F
        1. Z 归一化到 mu=0, std=1
        2. *不*归一化到 mu=0, std=1
    - y = f(X)
        1. 移除 NaN
        2. 移除离群太远的值
        3. *不*调整 f 使得 X 和 y 具有 mu=0, std=1
    return [X, y], f
    """
    def __init__(self, args:AttrDict, beyond_token=False):
        super().__init__(args, beyond_token=beyond_token)
        self.data_generator = SubeqGenerator(SNIPGenerator(**args), max_value=self.tokenizer.MAX_VALUE, max_var=args.max_var, normalize_X=False)

    def __getitem__(self, idx):
        while True:
            eqtree = self.eqtree_generator.generate_eqtree()

            N = self.args.uniform_sample_number or np.random.randint(100, 500)
            X, Y = self.data_generator.generate_data(int(1.2*N), eqtree)
            if X is None: continue

            if self.args.get('plot_demo', False):
                fi, fig, axes = get_fig(2, 2, AW=3)

                if X is None: continue
                if X.shape[1] > 2:    axes[0].scatter(*X[:, (0,1)].T, c=Y, cmap=plt.cm.bwr, s=1)
                elif X.shape[1] == 2: axes[0].scatter(*X.T, c=Y, cmap=plt.cm.bwr, s=1)
                elif X.shape[1] == 1: axes[0].scatter(X, Y, c=Y, cmap=plt.cm.bwr, s=1)

            dis = np.square(Y - np.mean(Y)) + np.square(X - np.mean(X, axis=0)).sum(axis=1)
            X, Y = X[dis.argsort()[:N]], Y[dis.argsort()[:N]]

            if self.args.get('plot_demo', False):
                if X.shape[1] > 2:    axes[1].scatter(*X[:, (0,1)].T, c=Y, cmap=plt.cm.bwr, s=1)
                elif X.shape[1] == 2: axes[1].scatter(*X.T, c=Y, cmap=plt.cm.bwr, s=1)
                elif X.shape[1] == 1: axes[1].scatter(X, Y, c=Y, cmap=plt.cm.bwr, s=1)

            variables = sorted(set([x.name for x in eqtree.iter_preorder() if isinstance(x, nd.Variable)]))
            if self.args.normalize_X: 
                X_mean = np.mean(X, axis=0)
                X_std = np.std(X, axis=0).clip(1e-6)
                X = (X - X_mean) / X_std
                # tmp = {var:[] for var in variables}
                # for x in eqtree.iter_preorder():
                #     if isinstance(x, nd.Variable):
                #         tmp[x.name].append(x)
                # for idx, var in enumerate(variables):
                #     for x in tmp[var]:
                #         x.replace(nd.Variable(var) * X_std[idx] + X_mean[idx])
            if self.args.normalize_y: 
                Y_mean = np.mean(Y)
                Y_std = np.std(Y).clip(1e-6)
                Y = (Y - Y_mean) / Y_std
                # eqtree = (eqtree - Y_mean) / Y_std

            if self.args.simplify:
                sympy_expr = str2sympy(str(eqtree))
                sympy_expr = sympy_expr.xreplace({n: round(n, 7) for n in sympy_expr.atoms(sp.Number)})
                sympy_expr = sp.nsimplify(sympy_expr)
                sympy_expr = sympy_expr.subs({sym: sp.Symbol(sym.name, real=True) for sym in sympy_expr.free_symbols})
                try:
                    _eqtree = sympy2eqtree(sympy_expr)
                    if self.args.save_equations:
                        with open(os.path.join(self.args.save_dir, 'equations.txt'), 'a') as f:
                            f.write(f'[{len(eqtree)}->{len(_eqtree)}]\t{eqtree.to_str(number_format=".2f")}\t{_eqtree.to_str(number_format=".2f")}\n')
                        self.args.save_equations -= 1
                    if len(_eqtree) <= len(eqtree): eqtree = _eqtree
                except:
                    continue

            if not any(isinstance(x, nd.Variable) for x in eqtree.iter_preorder()): continue
            if len(eqtree) > self.args.max_len: continue

            if self.args.get('plot_demo', False):
                norm = mcolors.Normalize(vmin=Y.min(), vmax=Y.max())
                if X.shape[1] > 2:    axes[2].scatter(*X[:, (0,1)].T, c=Y, cmap=plt.cm.bwr, norm=norm, s=1)
                elif X.shape[1] == 2: axes[2].scatter(*X.T, c=Y, cmap=plt.cm.bwr, norm=norm, s=1)
                elif X.shape[1] == 1: axes[2].scatter(X, Y, c=Y, cmap=plt.cm.bwr, norm=norm, s=1)

                variables = sorted(set(x.name for x in eqtree.iter_preorder() if isinstance(x, nd.Variable)))
                X_ = np.random.normal(0, 1, (10000, X.shape[1]))
                Y_ = eqtree.eval({var: X_[:, idx] for idx, var in enumerate(variables)})
                if X_.shape[1] > 2:    axes[3].scatter(*X_[:, (0,1)].T, c=Y_, cmap=plt.cm.bwr, norm=norm, s=1)
                elif X_.shape[1] == 2: axes[3].scatter(*X_.T, c=Y_, cmap=plt.cm.bwr, norm=norm, s=1)
                elif X_.shape[1] == 1: axes[3].scatter(X_, Y_, c=Y_, cmap=plt.cm.bwr, norm=norm, s=1)
                axes[2].set_xlim(*axes[3].get_xlim())
                axes[2].set_ylim(*axes[3].get_ylim())
            
                fig.suptitle(f'${eqtree.to_str(latex=True, number_format=".2f", omit_mul_sign=True)}$')
                os.makedirs('./plot/demo', exist_ok=True)
                fig.savefig(f'./plot/demo/{eqtree.to_str(number_format=".1f").replace("/", "div").replace(" ", "")}.png')
                fig.savefig(f'./test.png')

            data = np.pad(X, ((0, 0), (0, self.args.max_var - X.shape[1])), constant_values=0)
            data = np.concatenate([data, Y[:, None]], axis=1)
            length = len(eqtree)

            used_vars = [1] * X.shape[1] + [0] * (self.args.max_var - X.shape[1])

            if not self.beoynd_token:
                returns = self.tokenizer.float2token(data), self.tokenizer.eqtree2token(eqtree), used_vars, length
            returns = self.tokenizer.float2index(data), self.tokenizer.eqtree2index(eqtree), used_vars, length
            return returns


class Num2EqDatasetPure(Num2EqDataset):
    """
    生成的公式没有数值常数，只有符号，因此称为 Pure, 另外 X 是 Hard 的
    - f ~ F'
        1. 生成 N 个变量和 M 个二元运算符的树 
        2. 向树中插入一元运算符 
        3. *不*在可能的地方插入数值常数 
        4. (optional) 化简
    - X = phi(Z), Z ~ GMM(K), phi_{1..D} ~ F (同 Hard)
        1. Z 归一化到 mu=0, std=1
        2. *不*归一化到 mu=0, std=1
    - y = f(X)
        1. 移除 NaN
        2. (optional) 归一化到 mu=0, std=1
        3. *不*调整 f 使得 X 和 y 具有 mu=0, std=1
    return [X, y], f
    """
    def __init__(self, args:AttrDict, beyond_token=False):
        super().__init__(args, beyond_token=beyond_token)
        self.data_generator = SubeqGenerator(SNIPGenerator(**args), max_value=self.tokenizer.MAX_VALUE, max_var=args.max_var, normalize_X=False)
        self.eqtree_generator = SNIPGenerator3(**args)

    def __getitem__(self, idx):
        while True:
            eqtree = self.eqtree_generator.generate_eqtree()
            if self.args.simplify:
                sympy_expr = str2sympy(str(eqtree))
                sympy_expr = sympy_expr.subs({sym: sp.Symbol(sym.name, real=True) for sym in sympy_expr.free_symbols})
                try:
                    _eqtree = sympy2eqtree(sympy_expr)
                    if self.args.save_equations:
                        with open(os.path.join(self.args.save_dir, 'equations.txt'), 'a') as f:
                            f.write(f'[{len(eqtree)}->{len(_eqtree)}]\t{eqtree.to_str(number_format=".2f")}\t{_eqtree.to_str(number_format=".2f")}\n')
                        self.args.save_equations -= 1
                    if len(_eqtree) <= len(eqtree): eqtree = _eqtree
                except:
                    continue
            length = len(eqtree)
            if not any(isinstance(x, nd.Variable) for x in eqtree.iter_preorder()): continue

            N = self.args.uniform_sample_number or np.random.randint(100, 500)
            X, Y = self.data_generator.generate_data(N, eqtree)
            if X is None: continue
            if self.args.normalize_X: X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
            if self.args.normalize_y: Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-6)

            if not any(isinstance(x, nd.Variable) for x in eqtree.iter_preorder()): continue
            if len(eqtree) > self.args.max_len: continue

            data = np.pad(X, ((0, 0), (0, self.args.max_var - X.shape[1])), constant_values=0)
            data = np.concatenate([data, Y[:, None]], axis=1)

            used_vars = [1] * X.shape[1] + [0] * (self.args.max_var - X.shape[1])

            if not self.beoynd_token:
                return self.tokenizer.float2token(data), self.tokenizer.eqtree2token(eqtree), used_vars, length
            return self.tokenizer.float2index(data), self.tokenizer.eqtree2index(eqtree), used_vars, length


class Num2EqDatasetKeep(Num2EqDataset):
    """
    生成 [x_1, x_2, ..., x_D, f_1, f_2, ..., f_{K-D}] vs f(x_1, x_2, ..., x_D) 的数据
    f_1, f_2, ..., f_{K-D} 是 x_1, x_2, ..., x_D 的函数
    f_1, f_2, ..., f_{K-D} 是 f 的子函数，从 f 的公式树上随机选择
    公式长度定义为 f(x_1, x_2, ..., x_D, f_1, f_2, ..., f_{K-D}) 的长度, 显然它不会大于 f(x_1, x_2, ..., x_D) 的长度
    为了提升鲁棒性， f_1, f_2, ..., f_{K-D} 中会有一些随机生成、不在 f 的公式树上的函数
    这时可能需要验证随机生成的它们确实不在 f 的公式树上。但是感觉太麻烦了所以算了。这可能导致模型倾向于高估公式长度？
    """
    def __init__(self, args:AttrDict, beyond_token=False):
        super().__init__(args, beyond_token=beyond_token)
        self.eqtree_generator.min_unary *= 3
        self.eqtree_generator.max_unary *= 3
        self.eqtree_generator.max_unary_depth *= 2
        self.eqtree_generator.min_binary_per_var = 1
        self.eqtree_generator.max_binary_per_var = 3
        self.eqtree_generator.max_var = 5

    def __getitem__(self, idx):
        while True:
            eqtree:nd.Symbol = self.eqtree_generator.generate_eqtree()
            if self.args.simplify:
                sympy_expr = str2sympy(str(eqtree))
                sympy_expr = sympy_expr.subs({sym: sp.Symbol(sym.name, real=True) for sym in sympy_expr.free_symbols})
                # sympy_expr = sp.simplify(sympy_expr)
                try:
                    _eqtree = sympy2eqtree(sympy_expr)
                except:
                    continue
                if len(_eqtree) <= len(eqtree): eqtree = _eqtree
            if not any(isinstance(x, nd.Variable) for x in eqtree.iter_preorder()): continue

            N = self.args.uniform_sample_number or np.random.randint(100, 500)
            X_dict, Y = self.data_generator.generate_data(N, eqtree, return_X_dict=True)
            if X_dict is None: continue
            if self.args.normalize_y: Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-6)

            variables = list(set(x.name for x in eqtree.iter_preorder() if isinstance(x, nd.Variable)))
            L = len(eqtree)
            D = len(variables)
            K = random.randint(D, min(self.args.max_var, L))
            F = {}
            for i in range(D+1, K+1):
                choices = [x for x in eqtree.iter_preorder() if not isinstance(x, (nd.Number, nd.Variable))]
                prob = np.array([1/len(c) for c in choices])
                f:nd.Symbol = np.random.choice(choices, p=prob/prob.sum())
                F[f'x_{i}'] = f
                X_dict[f'x_{i}'] = f.eval(X_dict) + 0 * Y  # 0 * Y 用于保持 Y 的 shape
                L -= len(f) - 1
                if f == eqtree: 
                    eqtree = nd.Variable(f'x_{i}')
                    break
                f.replace(nd.Variable(f'x_{i}'))
            if len(eqtree) != L: # 可能是哪里有 bug
                logger.warning(f'L={L}, len(eqtree)={len(eqtree)}, eqtree={eqtree}, F={F}')
            if len(eqtree) > self.args.max_len: continue
            
            X = np.stack(list(X_dict.values()), axis=-1)
            if self.args.normalize_X: X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
            data = np.pad(X, ((0, 0), (0, self.args.max_var - X.shape[1])), constant_values=0)
            data = np.concatenate([data, Y[:, None]], axis=1)

            variables = [var.name for var in eqtree.iter_preorder() if isinstance(var, nd.Variable)]
            used_vars = [int(var in variables) for var in X_dict.keys()] + [0] * (self.args.max_var - len(X_dict))

            if self.args.save_equations > 0:
                with open(os.path.join(self.args.save_dir, 'equations.txt'), 'a') as f:
                    f.write(f'{L:2}\t{eqtree.to_str(omit_mul_sign=True, number_format=".2g")}, where ' + \
                            ', '.join(f'{k}={v.to_str(omit_mul_sign=True, number_format=".2g")}' for k, v in F.items()) + \
                            str(used_vars) + '\n')
                self.args.save_equations -= 1

            if not self.beoynd_token:
                return self.tokenizer.float2token(data), self.tokenizer.eqtree2token(eqtree), used_vars, L
            return self.tokenizer.float2index(data), self.tokenizer.eqtree2index(eqtree), used_vars, L


## to ignore
# /data3/yuzihan/WorkSpace/23-AutoAIFeynman/AutoAIFeynman/src/model/dataset/num2len.py:616: RuntimeWarning: Mean of empty slice
#   if True or self.args.normalize_X: data = (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-6)
# /data3/yuzihan/.conda/envs/yumeow/lib/python3.12/site-packages/numpy/lib/nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
#   var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")

class Num2EqDatasetLoad(Num2EqDataset):
    def __init__(self, eqtrees:List[nd.Symbol], args:AttrDict, beyond_token=False):
        super().__init__(args, beyond_token=beyond_token)
        self.eqtrees = eqtrees
        self.eqtree_generator = SimpleGenerator()
    
    def __getitem__(self, idx):
        while True:
            eqtree = random.choice(self.eqtrees).copy()
            variables = list(set([var.name for var in eqtree.iter_preorder() if isinstance(var, nd.Variable)]))
            mapping = {var: f'x_{i}' for i, var in enumerate(variables, 1)}
            for var in eqtree.iter_preorder():
                if isinstance(var, nd.Variable):
                    var.name = mapping[var.name]
            variables = list(set([var for var in eqtree.iter_preorder() if isinstance(var, nd.Variable)]))

            if self.args.simplify:
                sympy_expr = str2sympy(str(eqtree))
                sympy_expr = sympy_expr.xreplace({n: round(n, 7) for n in sympy_expr.atoms(sp.Number)})
                sympy_expr = sp.nsimplify(sympy_expr)
                sympy_expr = sympy_expr.subs({sym: sp.Symbol(sym.name, real=True) for sym in sympy_expr.free_symbols})
                try:
                    _eqtree = sympy2eqtree(sympy_expr)
                    if len(_eqtree) <= len(eqtree): eqtree = _eqtree
                except:
                    continue
            if not any(isinstance(x, nd.Variable) for x in eqtree.iter_preorder()): continue

            N = self.args.uniform_sample_number or np.random.randint(100, 500)
            Z, Y = self.data_generator.generate_data(N, eqtree, return_X_dict=True)
            if Z is None: continue

            x_list, f = random.choice(random.choice(list(islice(decompose(eqtree), 0, 1000))))
            raw_x_list = [x.copy() for x in x_list]
            x_list = list(set(x_list))
            x_list = list(filter(lambda x: not isinstance(x, nd.Number), x_list))
            if True or self.args.keep_vars: x_list = variables + list(set(x_list) - set(variables))
            used_vars = [1] * len(x_list)

            if np.random.rand() < 0.5: # 正例
                pass
            else: # 负例
                self.eqtree_generator.max_var = len(x_list)
                L = random.randint(1, 5)
                F = self.eqtree_generator.generate_eqtree(L)
                for var in F.iter_preorder():
                    if isinstance(var, nd.Variable):
                        idx = int(var.name.split('_')[1])
                        F = F.replace(var, x_list[idx-1])
                idx = random.randint(len(variables), len(x_list))
                x_list.insert(idx, F)
                used_vars.insert(idx, 0)
            if len(x_list) > self.args.max_var: continue

            self.eqtree_generator.max_var = len(variables)
            D_n = random.randint(0, self.args.max_var-len(x_list))
            for _ in range(D_n):
                L = random.randint(1, 5)
                x_list.append(self.eqtree_generator.generate_eqtree(L))
            used_vars += [0] * D_n

            for var in f.iter_preorder():
                if isinstance(var, nd.Empty):
                    x = raw_x_list.pop(0)
                    x = x if isinstance(x, nd.Number) else nd.Variable(f'x_{x_list.index(x)+1}')
                    f = f.replace(var, x)
            assert len(raw_x_list) == 0
            length = len(f)
            if length > self.args.max_len: continue

            data = np.stack([x.eval(Z)+0*Y for x in x_list], axis=-1)
            data = np.pad(data, ((0, 0), (0, self.args.max_var-data.shape[-1])), constant_values=0)
            if self.args.normalize_y: Y = (Y - Y.mean()) / (Y.std() + 1e-6)
            data = np.concatenate([data, Y[:, None]], axis=1)
            data[data == np.inf] = self.tokenizer.MAX_VALUE
            data[data == -np.inf] = -self.tokenizer.MAX_VALUE
            # if True or self.args.remove_abnormal: data[~((data > -5) & (data < 5)).all(axis=-1), :] = np.nan # NEED DEBUG
            if True or self.args.normalize_X: data = (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-6)
            assert np.isfinite(data[:, np.nonzero(used_vars)[0].tolist() + [-1]]).all()

            used_vars += [0] * (self.args.max_var - len(used_vars))
            if self.args.save_equations:
                with open(os.path.join(self.args.save_dir, 'equations.txt'), 'a') as file:
                    file.write(f'[{len(f)}]' + '\t' + \
                                f.to_str(number_format=".2f") + '\t' + \
                                '[' + ' | '.join(x.to_str(number_format=".2f") for x in x_list) + '\n' + ']' + '\t' + \
                                '[' + ' | '.join(str(x) for x in used_vars) + ']' + '\t' + \
                                eqtree.to_str(number_format=".2f") + '\n')
                self.args.save_equations -= 1

            if not self.beoynd_token:
                returns = self.tokenizer.float2token(data), self.tokenizer.eqtree2token(f), used_vars, length
            returns = self.tokenizer.float2index(data), self.tokenizer.eqtree2index(f), used_vars, length
            return returns


class InfiniteSampler(D.Sampler):
    # 无限生成索引，用于 DataLoader(Num2LenDataset(), sampler=InfiniteSampler())
    def __iter__(self):
        return itertools.count()
