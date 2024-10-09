import numpy as np
import sympy as sp
from typing import List, Generator, Tuple
from .symbols import *


def prefix2eqtree(prefix:List[str]):
    def foo(idx):
        item, idx = eval(prefix[idx]), idx+1
        operands = []
        if type(item) == int: 
            operands, item = [item], Number
    
        for _ in range(item.n_operands):
            operand, idx = foo(idx)
            operands.append(operand)
        return item(*operands), idx

    eqtree, idx = foo(0)
    assert idx == len(prefix)
    return eqtree


_sympy2symbol:Dict[str,Symbol] = {
    'Add': Add,
    'Mul': Mul,
    'Pow': Pow,
    'Abs': Abs,
    'sin': Sin,
    'cos': Cos,
    'tan': Tan,
    'exp': Exp,
    'log': Log,
    'tanh': Tanh,
    'arcsin': Arcsin,
    'arccos': Arccos,
    'arctan': Arctan,
    'cot': Cot,
}
def sympy2eqtree(expr:sp.Expr, merge_numbers=False):
    if expr.is_Atom:
        if expr.is_Symbol:
            return Variable(expr.name)
        elif expr.is_infinite:
            return Number(float('inf'))
        else:
            return Number(float(expr))
    symbol = _sympy2symbol[expr.func.__name__]
    operands = [sympy2eqtree(arg) for arg in expr.args]
    if merge_numbers and all(isinstance(op, Number) for op in operands):
        return Number(symbol.create_instance(*operands).eval())
    return symbol.create_instance(*operands)


def str2sympy(eq:str, variables=['alpha', 'beta', 'gamma', 'I']):
    return sp.parse_expr(eq, local_dict={var:sp.Symbol(var) for var in variables})


def simplify(eq:Symbol):
    try:
        variables = list(set(var.name for var in eq.preorder() if isinstance(var, Variable)))
        return sympy2eqtree(sp.simplify(str2sympy(str(eq), variables)), merge_numbers=True)
    except:
        return eq.copy()


def decompose(eqtrees:Symbol|List[Symbol], 
              route:List[List[Symbol]]=[],
              route2:List[Symbol]=[Empty()]) -> Generator[List[Tuple[List[Symbol], Symbol]], None, None]:
    """
    将 eqtrees 分解为 Leaf Symbol 的组合，尝试所有可能的分解方式，每次返回一种分解方式
    Input: exp(x+y)*(x+z)
    Output1:
        [exp(x + y) * (x + z)] | ?
        [exp(x + y), x + z] | ? * ?
        [x + y, x + z] | exp(?) * ?
        [x, y, x + z] | exp(? + ?) * ?
        [x, y, x, z] | exp(? + ?) * (? + ?)
    Output2:
        [exp(x + y) * (x + z)] | ?
        [exp(x + y), x + z] | ? * ?
        [x + y, x + z] | exp(?) * ?
        [x + y, x, z] | exp(?) * (? + ?)
        [x, y, x, z] | exp(? + ?) * (? + ?)
    Output3:
        [exp(x + y) * (x + z)] | ?
        [exp(x + y), x + z] | ? * ?
        [exp(x + y), x, z] | ? * (? + ?)
        [x + y, x, z] | exp(?) * (? + ?)
        [x, y, x, z] | exp(? + ?) * (? + ?)
    """
    if isinstance(eqtrees, Symbol): eqtrees = [eqtrees]
    if all([eq.n_operands == 0 for eq in eqtrees]): 
        yield list(zip([*route, eqtrees], route2))
    else:
        for idx, eq in enumerate(eqtrees):
            if eq.n_operands == 0: continue
            f = route2[-1].copy()
            cnt = -1
            for n in f.preorder():
                if isinstance(n, Empty): 
                    cnt +=1
                    if cnt == idx: 
                        if n.parent: n.replace(eq.__class__())
                        else: f = eq.__class__()
                        break
            else:
                raise Exception('Error')
            yield from decompose([*eqtrees[:idx], *eq.operands, *eqtrees[idx+1:]], 
                                 [*route, eqtrees],
                                 [*route2, f])
