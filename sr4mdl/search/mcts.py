import json
import time
import torch
import random
import logging
import sklearn
import traceback
import numpy as np
import sympy as sp
import nd2py as nd
import pandas as pd
from tqdm import tqdm
from typing import List, Generator, Tuple, Dict
from ..env import simplify
# from ..env.symbols import *
from ..env.tokenizer import Tokenizer
from ..model.mdlformer import MDLformer
from .utils import preprocess, sample_Xy
from nd2py.utils import seed_all, Timer, NamedTimer, R2_score, RMSE_score

class Node:
    def __init__(self, eqtrees:List[nd.Symbol]):
        # Formula part
        self.eqtrees = eqtrees
        self.phi = None
        self.complexity = None
        self.reward = None
        self.r2 = None

        # MC Tree part
        self.parent = None
        self.children = []
        self.N = 0
        self.Q = 0
        self.MDL = None
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '[' + ', '.join(str(eq) for eq in self.eqtrees) + ']' + f' (N={self.N}, Q={self.Q/(self.N+1e-6):.2f}, MDL={self.MDL:.0f})'
    
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
                UCT = {x: x.Q/(x.N+1e-6) + c / max(x.MDL, 1e-6) * np.sqrt(node.N) / (x.N+1) for x in siblings}
                siblings = sorted(siblings, key=UCT.get, reverse=True)
                siblings = siblings[:N]
            else:
                siblings = [node]
                UCT = {node: 0.0}
            new_items = [f'{node} (MDL={node.MDL:.2f}, UCT={UCT[node]:.2f})' for node in siblings]
            self_idx = siblings.index(node)
            for idx, item in enumerate(items): 
                items[idx] = ('├ ' if idx < len(items)-1 else '└ ') + item.replace('\n', '\n' + ('┆ ' if idx < len(items)-1 else '  '))
            new_items[self_idx] = '\033[31m' + new_items[self_idx] + '\033[0m' + ('\n' if items else '') + '\n'.join(items)
            items = new_items
        assert len(items) == 1
        return items[0]

    def copy(self) -> 'Node':
        copy = Node([eqtree.copy() for eqtree in self.eqtrees])
        copy.phi = self.phi
        copy.complexity = self.complexity
        copy.reward = self.reward
        copy.r2 = self.r2
        copy.MDL = self.MDL
        return copy
        

class MCTS4MDL(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Monte Carlo Tree Search from the sample side, only use f but not g
    """
    def __init__(self, 
        tokenizer:Tokenizer, 
        model:MDLformer, 
        binary:List[str|nd.Symbol]=[nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Max, nd.Min],
        unary:List[str|nd.Symbol]=[nd.Sqrt, nd.Log, nd.Abs, nd.Neg, nd.Inv, nd.Sin, nd.Cos, nd.Tan],
        leaf:List[float|nd.Number]=[nd.Number(1), nd.Number(0.5)],
        const_range=None,
        child_num=50,
        n_playout=100,
        d_playout=10,
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
        **kwargs):
        self.tokenizer = tokenizer # Tokenizer(-100, 100, 4)
        self.model = model # DataEncoder(AttrDict({}), self.tokenizer.get_token_list(all=False))
        self.max_var = self.model.args.max_var

        self.eqtree = None
        self.binary = [eval(x, globals(), nd.__dict__) if isinstance(x, str) else x for x in binary]
        self.unary = [eval(x, globals(), nd.__dict__) if isinstance(x, str) else x for x in unary]
        self.leaf = [nd.Number(x) if isinstance(x, float) else x for x in leaf]
        self.variables = []

        self.const_range = const_range
        self.child_num = child_num
        self.n_playout = n_playout
        self.d_playout = d_playout
        self.max_len = max_len
        self.c = c
        self.n_iter = n_iter
        self.sample_num = sample_num

        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.records = []
        self.logger = logging.getLogger(__name__)
        self.step_timer = Timer()
        self.view_timer = Timer()
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

    def fit(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray], y, n_iter=None, use_tqdm=False, 
            early_stop:callable=lambda r2, complexity, eq: r2 > 0.99999):
        """
        Args:
            X: (n_samples, n_dims)
            y: (n_samples,)
        """
        seed_all(self.random_state)
        n_iter = n_iter or self.n_iter

        # Preprocess
        X = preprocess(X)
        X, y = sample_Xy(X, y, self.sample_num)
        self.variables = [nd.Variable(var) for var in X]
        self.keep_vars = len(X) if self.keep_vars and len(X) <= self.max_var / 2 else 0

        # Root Node
        variables = list(X.keys())
        if len(variables) > self.max_var: variables = variables[:self.max_var]
        self.MC_tree = Node([nd.Variable(var) for var in variables])
        self.estimate_MDL(self.MC_tree, X, y)

        # Search
        stop = False
        self.best = None
        self.start_time = time.time()
        for iter in tqdm(range(1, n_iter+1), disable=not use_tqdm):
            record = {'iter': iter, 'time': time.time() - self.start_time}
            log = {'Iter': iter}

            leaf = self.select(self.MC_tree)
            expand = self.expand(leaf, X, y)
            reward, best_simulated = self.simulate(expand, X, y)
            self.backpropagate(expand, reward)
            
            self.step_timer.add(1)

            if self.best is None or best_simulated.reward > self.best.reward:
                self.best = best_simulated.copy()
                self.set_reward(self.best, X, y)
                self.eqtree = simplify(self.best.phi)
                record['complexity'] = self.best.complexity
                record['reward'] = self.best.reward
                record['r2'] = self.best.r2
                record['MDL'] = self.best.MDL
                record['eqtree'] = str(self.best)
                stop = early_stop(self.best.r2, self.best.complexity, self.best.phi)

            if (not iter % self.log_per_iter) or (self.step_timer.time > self.log_per_sec) or (iter == n_iter) or (stop):
                record['speed'] = (str(self.step_timer), str(self.view_timer))
                record['detailed_time'] = str(self.named_timer)
                
                log['Reward'] = f'{self.best.reward:.5f}'
                log['Complexity'] = self.best.complexity
                log['R2'] = f'{self.best.r2:.5f}'
                log['MDL'] = f'{self.best.MDL:.5f}'
                log['Best'] = str(self.best)
                log['Best equation'] = str(self.best.phi)
                log['Speed'] = f'{record['speed'][0]} ({record['speed'][1]})'
                log['Time'] = record['detailed_time']
                log['Current'] = str(expand)
                self.logger.info(' | '.join(f'\033[4m{k}\033[0m: {v}' for k, v in log.items()))

            self.records.append(record)
            if self.save_path:
                with open(self.save_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

            if stop:
                self.logger.note(f'Early stopping at iter {iter} with R2 {self.best.r2} ({self.best.eqtrees})')
                break
        # self.logger.info(expand.to_route(3, self.c))

    def predict(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray]) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_dims)
        Returns:
            y: (n_samples,)
        """
        if self.eqtree is None: raise ValueError('Model not fitted yet')
        X = preprocess(X)
        pred = self.eqtree.eval(X)
        pred[~np.isfinite(pred)] = 0
        return pred

    def action(self, state:Node, action:Tuple[nd.Symbol,int]) -> Node:
        """
        用 action[0] 取代 state.eqtrees[action[1]]
        """
        state = state.copy()
        eqtree, idx = action
        if idx == len(state.eqtrees): 
            state.eqtrees.append(eqtree)
        elif isinstance(eqtree, nd.Empty):
            state.eqtrees.pop(idx)
        else:
            state.eqtrees[idx] = eqtree
        return state

    def check_valid_action(self, state:Node, action:Tuple[nd.Symbol,int]) -> bool:
        eqtree, idx = action
        if idx > min(len(state.eqtrees) + 1, self.max_var): return False
        if idx == len(state.eqtrees) and isinstance(eqtree, nd.Empty): return False
        if len(state.eqtrees) == 1 and isinstance(eqtree, nd.Empty): return False
        if sum(len(eqtree) for i, eqtree in enumerate(state.eqtrees) if i != idx) + len(eqtree) > self.max_len: return False
        if idx < self.keep_vars: return False
        return True

    def iter_valid_action(self, state:Node, shuffle=False) -> Generator[Tuple[nd.Symbol,int],None,None]:
        leafs = [*state.eqtrees, *self.variables, *self.leaf]

        eqtree_loader = []
        for sym in self.binary:
            if sym in [nd.Add, nd.Mul, nd.Max, nd.Min]: # Abelian group
                for i in range(len(leafs)):
                    for j in range(i, len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
            elif sym in [nd.Sub, nd.Div]: # Non-abelian group
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
        for sym in self.variables:
            eqtree_loader.append(sym)
        eqtree_loader.append(nd.Empty())

        idx_loader = list(range(self.keep_vars, min(len(state.eqtrees) + 1, self.max_var)))

        loader = [(eqtree, idx) for eqtree in eqtree_loader for idx in idx_loader]
        if shuffle:
            random.shuffle(loader)

        for eqtree, idx in loader:
            if self.check_valid_action(state, (eqtree, idx)):
                yield eqtree, idx
    
    def pick_valid_action(self, state:Node) -> Tuple[nd.Symbol,int]:
        leafs = [*state.eqtrees, *self.variables, *self.leaf]
        for _ in range(1000):
            op = random.choice(self.binary + self.unary + self.variables + [nd.Empty()])
            idx = random.choice(range(self.keep_vars, min(len(state.eqtrees) + 1, self.max_var)))
            if isinstance(op, type): op = op(*random.choices(leafs, k=op.n_operands))
            if self.check_valid_action(state, (op, idx)): break
        else:
            raise ValueError('Cannot find valid action')
        return op, idx

    def estimate_MDL(self, nodes:Node|List[Node], X:Dict[str,np.ndarray], y:np.ndarray, batch_size=64) -> float:
        if isinstance(nodes, Node): nodes = [nodes]
        batch = np.zeros((len(nodes), y.shape[0], self.max_var+1)) # (B, N_i, D_max+1,)
        for idx, node in enumerate(nodes):
            for i, eqtree in enumerate(node.eqtrees): 
                batch[idx, :, i] = eqtree.eval(X)
        if self.normalize_y: y = (y - y.mean()) / (y.std()+1e-6)
        batch[:, :, -1] = y[np.newaxis, :]
        if self.remove_abnormal: batch[((batch > -5) & (batch < 5)).all(axis=-1), :] = np.nan
        if self.normalize_all: batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True)+1e-6)
        x = torch.from_numpy(self.tokenizer.float2index(batch, nan_to_pad=True)).to(self.model.device)
        pred = []
        for i in range(0, len(nodes), batch_size):
            pred.extend(self.model.predict(x[i:i+batch_size]).cpu().tolist())
        for node, c in zip(nodes, pred):
            node.MDL = c
        return pred if len(nodes) > 1 else pred[0]

    def select(self, root:Node) -> Node:
        node = root
        while node.children:
            node = max(node.children, key=lambda x: x.Q/(x.N+1e-6) + self.c / max(x.MDL, 1e-6) * np.sqrt(node.N) / (x.N+1))
        return node

    def expand(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Node:
        for idx, action in enumerate(self.iter_valid_action(node, shuffle=True)):
            child = self.action(node, action)
            child.parent = node
            child.xchild = len(node.children)
            node.children.append(child)
            if self.child_num and idx + 1 >= self.child_num: break
        if not node.children: return node  # leaf node
        self.estimate_MDL(node.children, X, y)
        return random.choice(node.children)

    def simulate(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Tuple[Node, float]:
        self.set_reward(node, X, y)
        best = node
        for i in range(self.n_playout):
            state = node
            for j in range(self.d_playout):
                action = self.pick_valid_action(state)
                if action is None: break
                state = self.action(state, action)
                self.set_reward(state, X, y)
                if state.reward > best.reward: best = state
        return best.reward, best

    def backpropagate(self, node:Node, reward:float):
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent

    def set_reward(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> float:
        self.view_timer.add(1)

        if self.train_eval_split < 1.0:
            train_idx = np.random.rand(y.shape[0]) < self.train_eval_split
            eval_idx = ~train_idx
        else:
            train_idx = np.ones_like(y).astype(bool)
            eval_idx = train_idx

        # Calculate Z
        Z = np.zeros((y.shape[0], 1+len(node.eqtrees)))
        Z[:, 0] = 1.0
        for idx, eqtree in enumerate(node.eqtrees, 1):
            try:
                Z[:, idx] = eqtree.eval(X)
            except:
                Z[:, idx] = np.nan
        Z[~np.isfinite(Z)] = 0

        # linear model as phi
        try:
            assert np.isfinite(Z).all()
            A, _, _, _ = np.linalg.lstsq(Z[train_idx, :], y[train_idx], rcond=None)
            A = np.round(A, 6)
            node.r2 = R2_score(y[eval_idx], Z[eval_idx, :] @ A)
            node.phi = nd.Number(A[0]) if A[0] != 0 else None
            for a, op in zip(A[1:], node.eqtrees):
                if a == 0: pass
                elif a == 1: 
                    if node.phi is None: node.phi = op
                    else: node.phi += op
                elif a == -1:
                    if node.phi is None: node.phi = -op
                    else: node.phi -= op
                else: 
                    if node.phi is None: node.phi = nd.Number(a) * op
                    else: node.phi += nd.Number(a) * op
            if node.phi is None: node.phi = nd.Number(0.0)
            node.complexity = len(node.phi)
            node.reward = self.eta ** node.complexity / (2 - node.r2)
        except Exception as e:
            self.logger.warning(traceback.format_exc())
            node.r2 = -np.inf
            node.complexity = np.inf
            node.reward = 0.0


        # prod model as phi: y = phi(f1, f2, ...) = a0 * |f1|^a1 * |f2|^a2 * ...
        try:
            Z_ = np.log(np.abs(Z).clip(1e-10, None))
            Z_[:, 0] = 1.0
            y_ = np.log(np.abs(y).clip(1e-10, None))
            assert np.isfinite(Z_).all() and np.isfinite(y_).all()
            A, _, _, _ = np.linalg.lstsq(Z_[train_idx, :], y_[train_idx], rcond=None)
            A[0] = np.exp(A[0])
            A = np.round(A, 6)
            prod = 1
            for z, a in zip(Z[:, 1:].T, A[1:]):
                prod *= np.abs(z) ** a
            A[0] *= np.sign(np.median(y[train_idx] / (A[0] * prod[train_idx]).clip(1e-6)))
            r2 = R2_score(y[eval_idx], A[0] * prod[eval_idx])
            reward = self.eta ** node.complexity / (2 - r2)
            if reward > node.reward:
                node.r2 = r2
                node.reward = reward
                node.phi = nd.Number(A[0]) if A[0] != 1 else None
                for idx, (a, op) in enumerate(zip(A[1:], node.eqtrees), 1):
                    if (Z[idx]<0).any(): op = nd.Abs(op)
                    if a == 0: pass
                    elif a == 1: 
                        if node.phi is None: node.phi = op
                        else: node.phi *= op
                    elif a == -1: 
                        if node.phi is None: node.phi = nd.Inv(op)
                        else: node.phi /= op
                    elif a == 2:
                        if node.phi is None: node.phi = nd.Pow2(op)
                        else: node.phi *= nd.Pow2(op)
                    elif a == -2:
                        if node.phi is None: node.phi = nd.Inv(nd.Pow2(op))
                        else: node.phi /= nd.Pow2(op)
                    elif a == 3:
                        if node.phi is None: node.phi = nd.Pow3(op)
                        else: node.phi *= nd.Pow3(op)
                    elif a == -3:
                        if node.phi is None: node.phi = nd.Inv(nd.Pow3(op))
                        else: node.phi /= nd.Pow3(op)
                    elif a == 0.5:
                        if node.phi is None: node.phi = nd.Sqrt(op)
                        else: node.phi *= nd.Sqrt(op)
                    elif a == -0.5:
                        if node.phi is None: node.phi = nd.Inv(nd.Sqrt(op))
                        else: node.phi /= nd.Sqrt(op)
                    elif a > 0:
                        if node.phi is None: node.phi = op ** nd.Number(a)
                        else: node.phi *= op ** nd.Number(a)
                    elif a < 0:
                        if node.phi is None: node.phi = nd.Inv(op ** nd.Number(-a))
                        else: node.phi /= op ** nd.Number(-a)
                    else: raise ValueError(f'Unknown a: {a}')
                if node.phi is None: node.phi = nd.Number(1.0)
                node.complexity = len(node.phi)
        except Exception as e:
            self.logger.warning(traceback.format_exc())
            # logger.warning(str(e))
            node.r2 = -np.inf
            node.complexity = np.inf
            node.reward = 0.0
        
        if not np.isfinite(node.reward): node.reward = 0.0
