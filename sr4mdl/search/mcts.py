import json
import time
import torch
import random
import logging
import sklearn
import numpy as np
import sympy as sp
import pandas as pd
from tqdm import tqdm
from typing import List, Generator, Tuple, Dict
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from ..env import sympy2eqtree, str2sympy
from ..env.symbols import *
from ..utils import Timer, NamedTimer
from ..env.tokenizer import Tokenizer
from ..model.mdlformer import MDLformer
from .utils import preprocess_X, rename_variable
from ..utils import set_seed

class Node:
    def __init__(self, eqtrees:List[Symbol]):
        self.eqtrees = eqtrees

        self.complexity = None
        self.R2 = None
        self.reward = None

        self.parent = None
        self.children = []
        self.C = None
        self.Q = 0
        self.N = 0
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '[' + ', '.join(str(eqtree) for eqtree in self.eqtrees) + ']' + f' ({self.N}*{self.Q/(self.N+1e-6):.2f})'
    
    def __format__(self, format_spec:str):
        if format_spec == 'eqtrees':
            return '[' + ' | '.join(str(f) for f in self.eqtrees) + ']'
        if format_spec == 'subtree':
            if not self.children: 
                return self.__str__()
            children = [child.__format__(format_spec) for child in self.children]
            for idx, child in enumerate(children):
                children[idx] = ('├ ' if idx < len(children)-1 else '└ ') + child.replace('\n', '\n' + ('┆ ' if idx < len(children)-1 else '  '))
            return self.__str__() + '\n' + '\n'.join(children)
        else:
            return self.__str__()
    
    def to_route(self, N=5, c=1.41) -> str:
        """
        Root
        ├ Node1
        ┆ ├ self
        ┆ └ Node1-2
        └ Node2
        """
        rev_route = [self]
        tmp = self
        while tmp.parent:
            rev_route.append(tmp.parent)
            tmp = tmp.parent
        items = []
        for node in rev_route:
            if node.parent:
                siblings = node.parent.children
                UCT = {x: x.Q/(x.N+1e-6) + c / max(x.C, 1e-6) * np.sqrt(node.N) / (x.N+1) for x in siblings}
                siblings = sorted(siblings, key=UCT.get, reverse=True)
                siblings = siblings[:N]
            else:
                siblings = [node]
                UCT = {node: 0.0}
            new_items = [f'{node} (C={node.C:.2f}, UCT={UCT[node]:.2f})' for node in siblings]
            self_idx = siblings.index(node)
            for idx, item in enumerate(items): 
                items[idx] = ('├ ' if idx < len(items)-1 else '└ ') + item.replace('\n', '\n' + ('┆ ' if idx < len(items)-1 else '  '))
            new_items[self_idx] = '\033[31m' + new_items[self_idx] + '\033[0m' + ('\n' if items else '') + '\n'.join(items)
            items = new_items
        assert len(items) == 1
        return items[0]
        

    def fit(self, X:Dict[str,np.ndarray], y:np.ndarray):
        parameters = [op for op in self.preorder() if isinstance(op, Number)]
        if len(parameters) == 0: return self
        def loss(params):
            for idx, param in enumerate(params):
                parameters[idx].value = param
            return np.mean((y - self.eval(**X)) ** 2)
        x0 = [param.value for param in parameters]
        res = minimize(loss, x0, method='BFGS')
        for idx, param in enumerate(res.x):
            parameters[idx].value = param

    def copy(self) -> 'Node':
        copy = Node([eqtree.copy() for eqtree in self.eqtrees])
        copy.complexity = self.complexity
        copy.R2 = self.R2
        copy.reward = self.reward
        copy.C = self.C
        return copy
    
    def merge_eqtrees(self, X:Dict[str,np.ndarray], y:np.ndarray, simplified=False) -> Symbol:          
        eqtree:Symbol = self.phi
        # Z = []
        # for eqtree in self.eqtrees:
        #     Z.append(eqtree.eval(**X))
        #     if isinstance(Z[-1], numbers.Number): Z[-1] = np.full_like(y, Z[-1])
        # Z = np.stack(Z, axis=1)
        # Z[~np.isfinite(Z)] = 0
        # Z = np.pad(Z, ((0, 0), (1, 0)), constant_values=1)
        # A = np.linalg.lstsq(Z, y, rcond=None)[0]
        # A = np.round(A, 6)
        # eqtree = Number(A[0])
        # for a, op in zip(A[1:], self.eqtrees):
        #     if a == 0: continue
        #     elif a == 1: eqtree = Add(eqtree, op)
        #     elif a == -1: eqtree = Sub(eqtree, op)
        #     else: eqtree = Add(eqtree, Mul(Number(a), op))
        if simplified:
            try:
                eqtree = sympy2eqtree(sp.parse_expr(str(eqtree), local_dict={k: sp.Symbol(k) for k in X}))
            except Exception as e:
                logger.warning(f'Error in simplifying {eqtree}: {e}')
        return eqtree
        

class MCTS4MDL(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Monte Carlo Tree Search from the sample side, only use f but not g
    """
    def __init__(self, 
        tokenizer:Tokenizer, 
        model:MDLformer, 
        binary:List[str|Symbol]=[Add, Sub, Mul, Div, Max, Min],
        unary:List[str|Symbol]=[Sqrt, Log, Abs, Neg, Inv, Sin, Cos, Tan],
        leaf:List[float|Number]=[Number(1), Number(0.5)],
        const_range=(-1.0, 1.0),
        child_num=50,
        n_playout=100,
        max_len=30,
        c=1.41,
        n_iter=100,
        sample_num=300,
        log_per_iter=float('inf'),
        log_per_sec=float('inf'),
        save_path=None,
        keep_vars=False,
        normalize_y=False,
        normalize_all=False,
        remove_abnormal=False,
        random_state=42,
        train_eval_split=1.0,
        eta=0.999,
        **kwargs
        ):
        self.tokenizer = tokenizer # Tokenizer(-100, 100, 4)
        self.model = model # DataEncoder(AttrDict({}), self.tokenizer.get_token_list(all=False))
        self.max_var = self.model.args.max_var

        self.eqtree = None
        self.binary = [eval(x) if isinstance(x, str) else x for x in binary]
        self.unary = [eval(x) if isinstance(x, str) else x for x in unary]
        self.leaf = [Number(x) if isinstance(x, float) else x for x in leaf]
        self.terminal_op = []

        self.const_range = const_range
        self.child_num = child_num
        self.n_playout = n_playout
        self.max_len = max_len
        self.c = c
        self.n_iter = n_iter
        self.sample_num = sample_num

        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.records = []
        self.logger = logging.getLogger('my.SampleSideMCTS')
        self.speed_timer = Timer()
        self.named_timer = NamedTimer()
        self.save_path = save_path
        self.keep_vars = keep_vars
        self.normalize_y = normalize_y
        self.normalize_all = normalize_all
        self.remove_abnormal = remove_abnormal
        self.random_state = random_state
        self.train_eval_split = train_eval_split
        self.eta = eta

        if kwargs:
            self.logger.warning('Unknown args: %s', ', '.join(f'{k}={v}' for k,v in kwargs.items()))

    def __repr__(self):
        res = 'None' if self.eqtree is None else self.eqtree.to_str()
        return '{}({})'.format(self.__class__.__name__, res)

    def fit(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray], y, n_iter=None, use_tqdm=False):
        """
        Args:
            X: (n_samples, n_dims)
            y: (n_samples,)
        """
        set_seed(self.random_state)

        X, variable_mapping = preprocess_X(X)
        total_num = len(next(X.values().__iter__()))
        if self.sample_num is not None and total_num > self.sample_num:
            idx = np.random.choice(total_num, self.sample_num, replace=False)
            X = {k: v[idx] for k, v in X.items()}
            y = y[idx]

        self.terminal_op = [Variable(f'x_{i+1}') for i in range(len(X))]
        if self.const_range is not None: 
            self.terminal_op.append(Number(np.random.uniform(*self.const_range)))
        self.keep_vars_num = len(X) if self.keep_vars and (len(X) <= self.max_var / 2) else 0

        D = min(len(X), self.max_var)
        self.MC_tree = Node([Variable(f'x_{i+1}') for i in range(D)])
        self.estimate_C(self.MC_tree, X, y)
        self.best = None

        self.start_time = time.time()
        n_iter = n_iter or self.n_iter
        for iter in tqdm(range(1, n_iter+1), disable=not use_tqdm):
            record = {'iter': iter, 'time': time.time() - self.start_time}
            log = {'Iter': iter}

            leaf = self.select(self.MC_tree)
            expand = self.expand(leaf, X, y)
            best_simulated, reward = self.simulate(expand, X, y)
            self.backpropagate(expand, reward)
            
            self.speed_timer.add(1)

            if self.best is None or best_simulated.reward > self.best.reward:
                self.best = best_simulated.copy()
                self.set_reward(self.best, X, y)
                for eqtree in self.best.eqtrees:
                    for sym in eqtree.preorder():
                        if isinstance(sym, Variable):
                            sym.name = variable_mapping.get(sym.name, sym.name)
                self.eqtree = self.best.merge_eqtrees({v:X[k] for k, v in variable_mapping.items()}, y, simplified=True)
                record['reward'] = self.best.reward
                record['complexity'] = self.best.complexity
                record['r2'] = self.best.R2
                record['C'] = self.best.C
                record['eqtree'] = '{:eqtrees}'.format(self.best)

            if not iter % self.log_per_iter or self.named_timer.total() > self.log_per_sec or iter == n_iter:
                record['speed'] = self.speed_timer.pop()

                log['Reward'] = f'{self.best.reward:.5f}'
                log['Complexity'] = self.best.complexity
                log['R2'] = f'{self.best.R2:.5f}'
                log['C'] = f'{self.best.C:.5f}'
                log['Best equation'] = '{:eqtrees}'.format(self.best)
                log['Speed'] = f'{record['speed']:.2f} iter/s'
                # log['Time'] = str(self.named_timer)
                self.named_timer.pop()
                log['Current'] = expand.__format__('subtree')

                self.logger.info(' | '.join(f'\033[4m{k}\033[0m: {v}' for k, v in log.items()))

            self.records.append(record)
            if self.save_path:
                with open(self.save_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
            if self.best.R2 is not None and self.best.R2 > 0.99999:
                self.logger.info(f'Early stopping at iter {iter} with R2 {self.best.R2} ({self.best.eqtrees})') # + '\n' + expand.to_route(3, self.c))
                break

        return self

    def predict(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray]) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_dims)
        Returns:
            y: (n_samples,)
        """
        if self.eqtree is None:
            raise ValueError('Model not fitted yet')

        if isinstance(X, np.ndarray):
            X = {f'x_{i+1}': X[:, i] for i in range(X.shape[1])}
        elif isinstance(X, pd.DataFrame):
            X = {k:X[k].to_numpy() for k in X.columns}
        elif isinstance(X, dict):
            pass
        else:
            raise ValueError(f'Unknown type: {type(X)}')
        pred = self.eqtree.eval(**X)
        pred[~np.isfinite(pred)] = 0
        return pred

    def action(self, state:Node, action:Tuple[Symbol,int]) -> Node:
        """
        用 action[0] 取代 state.eqtrees[action[1]]
        """
        state = state.copy()
        eqtree, idx = action
        if idx == len(state.eqtrees): 
            state.eqtrees.append(eqtree)
        elif isinstance(eqtree, Empty):
            state.eqtrees.pop(idx)
        else:
            state.eqtrees[idx] = eqtree
        return state

    def check_valid_action(self, state:Node, action:Tuple[Symbol,int]) -> bool:
        eqtree, idx = action
        if idx > min(len(state.eqtrees) + 1, self.max_var): return False
        if idx == len(state.eqtrees) and isinstance(eqtree, Empty): return False
        if len(state.eqtrees) == 1 and isinstance(eqtree, Empty): return False
        if sum(len(eqtree) for i, eqtree in enumerate(state.eqtrees) if i != idx) + len(eqtree) > self.max_len: return False
        if idx < self.keep_vars_num: return False
        return True

    def iter_valid_action(self, state:Node, shuffle=False) -> Generator[Tuple[Symbol,int],None,None]:
        leafs = [*state.eqtrees, *self.terminal_op, *self.leaf]

        eqtree_loader = []
        for sym in self.binary:
            if sym in [Add, Mul, Max, Min]: # Abelian group
                for i in range(len(leafs)):
                    for j in range(i, len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
            elif sym in [Sub, Div]: # Non-abelian group
                for i in range(len(leafs)):
                    for j in range(len(leafs)):
                        if i != j:
                            eqtree_loader.append(sym(leafs[i], leafs[j]))
            else:
                for i in range(len(leafs)):
                    for j in range(len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
        for sym in self.unary:
            for i in range(len(leafs)):
                eqtree_loader.append(sym(leafs[i]))
        for sym in self.terminal_op:
            eqtree_loader.append(sym)
        eqtree_loader.append(Empty())

        idx_loader = list(range(self.keep_vars_num, min(len(state.eqtrees) + 1, self.max_var)))

        loader = [(eqtree, idx) for eqtree in eqtree_loader for idx in idx_loader]
        if shuffle:
            random.shuffle(loader)

        for eqtree, idx in loader:
            if self.check_valid_action(state, (eqtree, idx)):
                yield eqtree, idx
    
    def pick_valid_action(self, state:Node) -> Tuple[Symbol,int]:
        leafs = [*state.eqtrees, *self.terminal_op, Number(1)]
        while True:
            op = random.choice(self.binary + self.unary + self.terminal_op + [Empty()])
            idx = random.choice(range(self.keep_vars_num, min(len(state.eqtrees) + 1, self.max_var)))
            if isinstance(op, type): op = op(*random.choices(leafs, k=op.n_operands))
            if self.check_valid_action(state, (op, idx)): break
        return op, idx

    def estimate_C(self, nodes:Node|List[Node], X:Dict[str,np.ndarray], y:np.ndarray, batch_size=64) -> float:
        if isinstance(nodes, Node): nodes = [nodes]
        self.named_timer.add('drop')
        batch = np.zeros((len(nodes), y.shape[0], self.max_var+1)) # (B, N_i, D_max+1,)
        for idx, node in enumerate(nodes):
            for i, eqtree in enumerate(node.eqtrees): batch[idx, :, i] = eqtree.eval(**X)
        if self.normalize_y: y = (y - y.mean()) / (y.std()+1e-6)
        batch[:, :, -1] = y[np.newaxis, :]
        if self.remove_abnormal: batch[((batch > -5) & (batch < 5)).all(axis=-1), :] = np.nan
        if self.normalize_all: batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True)+1e-6)
        x = torch.from_numpy(self.tokenizer.float2index(batch, nan_to_pad=True)).to(self.model.device)
        self.named_timer.add('prepare batch')
        with torch.no_grad():
            pred = []
            for i in range(0, len(nodes), batch_size):
                pred.extend(self.model(x[i:i+batch_size]).cpu().tolist())
            # pred = self.model.predict(x).cpu().tolist()
        self.named_timer.add('forward')
        for node, c in zip(nodes, pred):
            node.C = c
        return pred if len(pred) > 1 else pred[0]

    def select(self, root:Node) -> Node:
        node = root
        # UCT = lambda x: x.Q/(x.N+1e-6) + self.c / (x.C).clip(1e-6) * np.sqrt(x.parent.N) / (x.N+1)
        while node.children:
            node = max(node.children, key=lambda x: x.Q/(x.N+1e-6) + self.c / max(x.C, 1e-6) * np.sqrt(node.N) / (x.N+1))
        return node

    def expand(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Node:
        for idx, action in enumerate(self.iter_valid_action(node, shuffle=True)):
            child = self.action(node, action)
            child.parent = node
            node.children.append(child)
            if self.child_num and idx + 1 >= self.child_num: break
        if node.children:
            self.estimate_C(node.children, X, y)
            return random.choice(node.children)
        else: # leaf node
            return node

    def simulate(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Tuple[Node, float]:
        node = node.copy()
        self.named_timer.add('drop')
        self.set_reward(node, X, y)
        self.named_timer.add('set_reward')
        best = node
        for _ in range(self.n_playout):
            state = node
            for __ in range(10):
                self.named_timer.add('drop')
                action = self.pick_valid_action(state)
                self.named_timer.add('pick-action')
                if action is None: break
                state = self.action(state, action)
                self.named_timer.add('drop')
                self.set_reward(state, X, y)
                self.named_timer.add('set_reward')
                if state.reward > best.reward: best = state
        return best, best.reward

    def backpropagate(self, node:Node, reward:float):
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent

    def set_reward(self, nodes:Node|List[Node], X:Dict[str,np.ndarray], y:np.ndarray) -> float:
        if self.train_eval_split < 1.0:
            train_idx = np.random.rand(y.shape[0]) < self.train_eval_split
            eval_idx = ~train_idx
        else:
            train_idx = np.ones_like(y).astype(bool)
            eval_idx = train_idx

        if isinstance(nodes, Node): nodes = [nodes]
        alpha = 1 / np.var(y[eval_idx]) / y[eval_idx].shape[0]
        for node in nodes:
            Z = np.zeros((y.shape[0], 1+len(node.eqtrees)))
            Z[:, 0] = 1.0
            for idx, eqtree in enumerate(node.eqtrees, 1):
                self.named_timer.add('drop')
                Z[:, idx] = eqtree.eval(**X)
                self.named_timer.add('set_reward: eval')
            node.complexity = sum(len(eqtree) + 1 for eqtree in node.eqtrees) - 1

            # phi = linear model
            try:
                # Z[~np.isfinite(Z)] = 0
                assert np.isfinite(Z).all()
                self.named_timer.add('set_reward: filter finite')
                A, residuals, _, _ = np.linalg.lstsq(Z[train_idx, :], y[train_idx], rcond=None)
                self.named_timer.add('set_reward: lstsq')
                residual = np.sum((y[eval_idx] - Z[eval_idx, :] @ A)**2)
                node.R2 = 1 - alpha * residual  # r2_score(y, y2) 比较慢
                node.reward = self.eta ** node.complexity / (2 - node.R2)

                A = np.round(A, 6)
                node.phi = Number(A[0])
                for a, op in zip(A[1:], node.eqtrees):
                    if a == 0: continue
                    elif a == 1: node.phi = Add(node.phi, op)
                    elif a == -1: node.phi = Sub(node.phi, op)
                    else: node.phi = Add(node.phi, Mul(Number(a), op))

            except Exception as e:
                # logger.warning(str(e))
                node.R2 = -np.inf
                node.reward = 0.0


            try:
                Z_ = np.log(np.abs(Z).clip(1e-10, None))
                Z_[:, 0] = 1.0
                y_ = np.log(np.abs(y).clip(1e-10, None))
                assert np.isfinite(Z_).all() and np.isfinite(y_).all()
                A, residuals, _, _ = np.linalg.lstsq(Z_[train_idx, :], y_[train_idx], rcond=None)
                self.named_timer.add('set_reward: lstsq')
                R2 = 1 - np.mean((y_[eval_idx] - Z_[eval_idx, :] @ A)**2) / np.var(y_[eval_idx])
                reward = self.eta ** node.complexity / (2 - R2)
                if reward > node.reward:
                    node.R2 = R2
                    node.reward = reward

                    A = np.round(A, 3)
                    node.phi = Number(np.exp(A[0]))
                    for a, op in zip(A[1:], node.eqtrees):
                        if a == 0: continue
                        elif a == 1: node.phi = Mul(node.phi, op)
                        elif a == -1: node.phi = Div(node.phi, op)
                        elif a == 2: node.phi = Mul(node.phi, Pow2(op))
                        elif a == -2: node.phi = Div(node.phi, Pow2(op))
                        elif a == 3: node.phi = Mul(node.phi, Pow3(op))
                        elif a == -3: node.phi = Div(node.phi, Pow3(op))
                        elif a > 0: node.phi = Mul(node.phi, Pow(op, Number(a)))
                        elif a < 0: node.phi = Div(node.phi, Pow(op, Number(-a)))
                        else: raise ValueError(f'Unknown a: {a}')
            except Exception as e:
                # logger.warning(str(e))
                node.R2 = -np.inf
                node.reward = 0.0
            
            if not np.isfinite(node.reward): node.reward = 0
