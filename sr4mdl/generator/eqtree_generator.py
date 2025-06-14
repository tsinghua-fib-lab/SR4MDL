import random
import logging
import numpy as np
import nd2py as nd
from typing import Tuple, List
from collections import defaultdict
from ..env import decompose

logger = logging.getLogger('EqtreeGenerator')

class SimpleGenerator:
    def __init__(self, max_param=2, max_unary=4, max_var=5, **kwargs):
        self.binary = [nd.Add, nd.Sub, nd.Mul, nd.Div]
        self.unary = [nd.Abs, nd.Inv, nd.Sqrt, nd.Log, nd.Exp, nd.Sin, nd.Arcsin, nd.Cos, nd.Arccos, nd.Tan, nd.Arctan, nd.Pow2, nd.Pow3]
        self.max_param = max_param
        self.max_unary = max_unary
        self.max_var = max_var

    def generate_eqtree(self, length:int) -> nd.Symbol:
        # n_binary_op + n_variable + n_parameter = length - n_unary_op
        # n_binary_op - n_variable - n_parameter = -1

        n_unary_op = np.random.randint(0, min(self.max_unary, length-1)+1) // 2 * 2 + (1 - length % 2)
        if not self.unary: n_unary_op = 0
        n_binary_op = (length - n_unary_op - 1) // 2
        n_parameter = np.random.randint(0, min(self.max_param, n_binary_op)+1)
        n_variable = n_binary_op + 1 - n_parameter
        variables = np.random.choice(np.random.randint(1, self.max_var+1), n_variable, replace=True)
        eqtrees = [nd.Variable(f"x_{i+1}") for i in variables] + [nd.Number(np.random.rand()) for i in range(n_parameter)]

        while n_unary_op or n_binary_op:
            if np.random.rand() < n_unary_op / (n_unary_op + n_binary_op):
                symbol = np.random.choice(self.unary)
                idx = np.random.choice(len(eqtrees))
                eqtrees[idx] = symbol(eqtrees[idx])
                n_unary_op -= 1
            else:
                symbol = np.random.choice(self.binary)
                idx1, idx2 = sorted(np.random.choice(len(eqtrees), 2, replace=False))
                op = eqtrees.pop(idx2)
                eqtrees[idx1] = symbol(eqtrees[idx1], op)
                n_binary_op -= 1
            
        assert len(eqtrees) == 1
        return eqtrees[0]


class GPLearnGenerator():
    def __init__(self, max_var=5, const_range:None|Tuple[float, float]=None, **kwargs):
        self.binary = kwargs.pop('binary', [nd.Add, nd.Sub, nd.Mul, nd.Div])
        self.unary = kwargs.pop('unary', [nd.Abs, nd.Inv, nd.Sqrt, nd.Log, nd.Exp, nd.Sin, nd.Arcsin, nd.Cos, nd.Arccos, nd.Tan, nd.Arctan, nd.Pow2, nd.Pow3])
        self.symbols = self.binary + self.unary
        self.full_prob = kwargs.pop('full_prob', 0.5)
        self.depth_range = kwargs.pop('depth_range', (2, 6))
        self.max_var = max_var
        self.const_range = const_range

        if any(kwargs):
            logger.warning(f"Unused arguments: {kwargs}")

    def generate_eqtree(self) -> nd.Symbol:
        full_tree = np.random.rand() < self.full_prob 
        max_depth = np.random.randint(*self.depth_range)
        op_prob = 1.0 if full_tree else 1 - self.max_var / (self.max_var + len(self.symbols))

        # Start a eqtree with a function to avoid degenerative eqtrees
        op = random.choice(self.symbols)
        eqtree = op()
        empty_nodes_and_depth = [(i, 1) for i in eqtree.operands]

        while empty_nodes_and_depth:
            empty_node, depth = empty_nodes_and_depth.pop(0)
            if (depth < max_depth) and (np.random.rand() < op_prob):
                op = random.choice(self.symbols)
                sym = op()
                eqtree = eqtree.replace(empty_node, sym)
                empty_nodes_and_depth.extend([(i, depth+1) for i in sym.operands])
            else: # Variable or Number
                sym = self.generate_leaf()
                eqtree = eqtree.replace(empty_node, sym)
        return eqtree
    
    def generate_leaf(self) -> nd.Number|nd.Variable:
        if self.const_range is not None:
            idx = np.random.randint(self.max_var + 1)
        else:
            idx = np.random.randint(self.max_var)
        if idx < self.max_var:
            return nd.Variable(f"x_{idx+1}")
        else:
            return nd.Number(np.random.uniform(*self.const_range))

class Sentinel(nd.Symbol):
    n_operands = 1
    def __init__(self, nettype='scalar'):
        super().__init__(nettype=nettype)

class MetaAIGenerator:
    def __init__(self, operators_to_downsample='Div:0,Arcsin:0,Arccos:0,Tan:0.2,Arctan:0.2,Sqrt:5,Pow2:3,Inv:3', **kwargs):
        self.binary = [nd.Add, nd.Sub, nd.Mul, nd.Div]
        self.unary = [nd.Abs, nd.Inv, nd.Sqrt, nd.Log, nd.Exp, nd.Sin, nd.Arcsin, nd.Cos, nd.Arccos, nd.Tan, nd.Arctan, nd.Pow2, nd.Pow3]
        
        prob_dict = defaultdict(lambda: 1.0)
        for item in operators_to_downsample.split(","):
            if item != "":
                op, prob = item.split(':')
                prob_dict[eval(op, globals(), nd.__dict__)] = float(prob)
        self.binary_prob = [prob_dict[op] for op in self.binary]
        self.binary_prob = np.array(self.binary_prob) / sum(self.binary_prob)
        self.unary_prob = [prob_dict[op] for op in self.unary]
        self.unary_prob = np.array(self.unary_prob) / sum(self.unary_prob)


    def generate_eqtree(self, n_operators, n_var) -> nd.Symbol:
        sentinel = Sentinel(); # 哨兵节点

        # construct unary-binary tree
        empty_nodes = [*sentinel.operands]
        next_en = -1
        n_empty = 1
        while n_operators > 0:
            next_pos, arity = self.generate_next_pos(n_empty, n_operators)
            op = self.generate_ops(arity)
            next_en += next_pos + 1
            n_empty -= next_pos + 1
            replace = empty_nodes[next_en]
            other = op()
            sentinel.replace(replace, other)
            empty_nodes.extend(other.operands)
            n_empty += op.n_operands
            n_operators -= 1
        
        # fill variables
        n_used_var = 0
        for n in empty_nodes:
            if n in sentinel.iter_preorder():
                sym, n_used_var = self.generate_leaf(n_var, n_used_var)
                sentinel = sentinel.replace(n, sym)

        return sentinel.operands[0]

    def dist(self, n_op, n_emp):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[n][e] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n - 1, e) + D(n - 1, e + 1)
        p1 = 0 if binary trees, or 1 if unary-binary trees
        """
        if not hasattr(self, 'dp_cache'): self.dp_cache = [[0]]
        p1 = 1 if self.unary else 0
        if len(self.dp_cache) <= n_op + n_emp:
            for _ in range(len(self.dp_cache), n_op + n_emp + 1):
                self.dp_cache[0].append(1)
                for r, row in enumerate(self.dp_cache[1:], 1):
                    row.append(row[-1] + p1 * self.dp_cache[r-1][-2] + self.dp_cache[r-1][-1])
                self.dp_cache.append([0])
        return self.dp_cache[n_op][n_emp]

    def generate_leaf(self, n_var:int, n_used_var:int) -> Tuple[nd.Symbol, int]:
        if n_used_var < n_var:
            return nd.Variable(f"x_{n_used_var+1}"), n_used_var+1
        else:
            idx = np.random.randint(1, n_var + 1)
            return nd.Variable(f"x_{idx}"), n_used_var

    def generate_ops(self, n_operands:int) -> nd.Symbol:
        if n_operands == 1:
            return np.random.choice(self.unary, p=self.unary_prob)
        else:
            return np.random.choice(self.binary, p=self.binary_prob)

    def generate_next_pos(self, n_empty, n_operators):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `n_empty` - 1}.
        """
        assert n_empty > 0
        assert n_operators > 0
        probs = [self.dist(n_operators - 1, n_empty - i + 1) for i in range(n_empty)]
        if self.unary:
            probs += [self.dist(n_operators - 1, n_empty - i) for i in range(n_empty)]
        probs = np.array(probs, dtype=np.float64) / self.dist(n_operators, n_empty)
        next_pos = np.random.choice(len(probs), p=probs)
        n_operands = 1 if next_pos >= n_empty else 2
        next_pos %= n_empty
        return next_pos, n_operands


class SNIPGenerator(MetaAIGenerator):
    def __init__(self, max_var=5, min_unary=0, max_unary=4, 
                 min_binary_per_var=0, max_binary_per_var=1, max_binary_ops_offset=4,
                 max_unary_depth=6, n_mantissa=4, max_exp=1, min_exp=0, **kwargs):
        self.max_var = max_var
        self.min_unary = min_unary
        self.max_unary = max_unary
        self.min_binary_per_var = min_binary_per_var
        self.max_binary_per_var = max_binary_per_var
        self.max_binary_ops_offset = max_binary_ops_offset
        self.max_unary_depth = max_unary_depth
        self.n_mantissa = n_mantissa # 4 -> 0.001 ~ 9.999
        self.max_exp = max_exp # max: 9.999 * 10^1
        self.min_exp = min_exp # min: 0.001 * 10^0
        super().__init__(**kwargs)

    def generate_eqtree(self, n_var=None, n_unary=None, n_binary=None) -> nd.Symbol:
        n_var = n_var or np.random.randint(1, self.max_var)
        n_unary = n_unary or np.random.randint(self.min_unary, self.max_unary+1)
        n_binary = n_binary or np.random.randint(self.min_binary_per_var * n_var, self.max_binary_per_var * n_var + self.max_binary_ops_offset)
        backup, self.unary = self.unary, []
        eqtree = super(SNIPGenerator, self).generate_eqtree(n_binary, n_var)
        self.unary = backup
        eqtree = self.add_unaries(eqtree, n_unary)
        eqtree = self.add_prefactors(eqtree)
        return eqtree

    def generate_float(self) -> nd.Number:
        sign = np.random.choice([-1, 1])
        mantissa = np.random.randint(1, 10 ** self.n_mantissa) / 10 ** (self.n_mantissa-1)
        exponent = np.random.randint(self.min_exp, self.max_exp)
        return nd.Number(sign * mantissa * 10 ** exponent)

    def _add_unaries(self, eqtree:nd.Symbol) -> nd.Symbol:
        for idx, op in enumerate(eqtree.operands):
            if len(op) < self.max_unary_depth:
                unary = np.random.choice(self.unary, p=self.unary_prob)
                eqtree.operands[idx] = unary(self._add_unaries(op))
            else:
                eqtree.operands[idx] = self._add_unaries(op)
        return eqtree

    def add_unaries(self, eqtree:nd.Symbol, n_unary:int) -> nd.Symbol:
        # Add some unary operations
        eqtree = self._add_unaries(eqtree)
        # Remove some unary operations
        postfix = [sym.__class__ if sym.n_operands > 0 else sym for sym in eqtree.iter_postorder()]
        indices = [i for i, x in enumerate(postfix) if x in self.unary]
        if len(indices) > n_unary:
            np.random.shuffle(indices)
            to_remove = indices[:len(indices) - n_unary]
            for index in sorted(to_remove, reverse=True):
                del postfix[index]
        eqtrees = []
        while postfix:
            sym = postfix.pop(0)
            if sym.n_operands == 0:
                eqtrees.append(sym)
            elif sym.n_operands == 1:
                eqtrees[-1] = sym(eqtrees[-1])
            elif sym.n_operands == 2:
                eqtrees[-2] = sym(eqtrees[-2], eqtrees[-1])
                eqtrees.pop(-1)
        assert len(eqtrees) == 1
        return eqtrees[0]
    
    def _add_prefactors(self, eqtree:nd.Symbol) -> nd.Symbol:
        if eqtree.__class__ in [nd.Add, nd.Sub]:
            x1, x2 = eqtree.operands
            if x1.__class__ not in [nd.Add, nd.Sub]:
                eqtree.operands[0] = nd.Mul(self.generate_float(), self._add_prefactors(x1))
            else:
                eqtree.operands[0] = self._add_prefactors(x1)
            eqtree.operands[0].parent = eqtree
            eqtree.operands[0].child_idx = 0
            if x2.__class__ not in [nd.Add, nd.Sub]:
                eqtree.operands[1] = nd.Mul(self.generate_float(), self._add_prefactors(x2))
            else:
                eqtree.operands[1] = self._add_prefactors(x2)
            eqtree.operands[1].parent = eqtree
            eqtree.operands[1].child_idx = 1
            return eqtree
        if eqtree.__class__ in self.unary and eqtree.operands[0].__class__ not in [nd.Add, nd.Sub]:
            a, b = self.generate_float(), self.generate_float()
            eqtree.operands[0] = nd.Add(a, nd.Mul(b, self._add_prefactors(eqtree.operands[0])))
            eqtree.operands[0].parent = eqtree
            eqtree.operands[0].child_idx = 0
            return eqtree
        for idx, op in enumerate(eqtree.operands):
            eqtree.operands[idx] = self._add_prefactors(op)
            eqtree.operands[idx].parent = eqtree
            eqtree.operands[idx].child_idx = idx
        return eqtree

    def add_prefactors(self, eqtree:nd.Symbol) -> nd.Symbol:
        _eqtree = self._add_prefactors(eqtree)
        if list(_eqtree.iter_preorder()) == list(eqtree.iter_preorder()):
            eqtree = nd.Mul(self.generate_float(), eqtree)
        eqtree = nd.Add(self.generate_float(), eqtree)
        return eqtree


class SNIPGenerator2(SNIPGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def add_prefactors(self, eqtree: nd.Symbol) -> nd.Symbol:
        variables = [op for op in eqtree.iter_preorder() if isinstance(op, nd.Variable)]
        for var in variables:
            a = self.generate_float()
            b = self.generate_float()
            eqtree = eqtree.replace(var, a * var.copy() + b)
        a = self.generate_float()
        b = self.generate_float()
        return a * eqtree + b


class SNIPGenerator3(SNIPGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_eqtree(self, n_var=None, n_unary=None, n_binary=None) -> nd.Symbol:
        n_var = n_var or np.random.randint(1, self.max_var)
        n_unary = n_unary or np.random.randint(self.min_unary, self.max_unary+1)
        n_binary = n_binary or np.random.randint(self.min_binary_per_var * n_var, self.max_binary_per_var * n_var + self.max_binary_ops_offset)
        backup, self.unary = self.unary, []
        eqtree = super(SNIPGenerator, self).generate_eqtree(n_binary, n_var)
        self.unary = backup
        eqtree = self.add_unaries(eqtree, n_unary)
        # eqtree = self.add_prefactors(eqtree)
        return eqtree

    def generate_float(self) -> nd.Number:
        raise NotImplementedError

class LoadedGenerator():
    def __init__(self, eqtrees:List[nd.Symbol], keep_vars=True):
        self.eqtrees = eqtrees
        self.keep_vars = keep_vars
    
    def generate_eqtree(self) -> nd.Symbol:
        eqtree = random.choice(self.eqtrees).copy()
        variables = sorted(set(x.name for x in eqtree.iter_preorder() if isinstance(x, nd.Variable)))

        vars, f = random.choice(list(decompose(eqtree)))
        vars = filter(lambda x: not isinstance(x, nd.Number), vars)
        
        z_list = list(set(vars))
        if self.keep_vars: z_list = variables + list(set(z_list) - set(variables))

