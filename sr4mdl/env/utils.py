import numpy as np
import sympy as sp
import nd2py as nd
from typing import List, Generator, Tuple, Dict


_sympy2symbol:Dict[str,nd.Symbol] = {
    'Add': nd.Add,
    'Mul': nd.Mul,
    'Pow': nd.Pow,
    'Abs': nd.Abs,
    'sin': nd.Sin,
    'cos': nd.Cos,
    'tan': nd.Tan,
    'exp': nd.Exp,
    'log': nd.Log,
    'tanh': nd.Tanh,
    'arcsin': nd.Arcsin,
    'arccos': nd.Arccos,
    'arctan': nd.Arctan,
    'cot': nd.Cot,
}
def sympy2eqtree(expr:sp.Expr, merge_numbers=False):
    if expr.is_Atom:
        if expr.is_Symbol:
            return nd.Variable(expr.name)
        elif expr.is_infinite:
            return nd.Number(float('inf'))
        else:
            return nd.Number(float(expr))
    symbol = _sympy2symbol[expr.func.__name__]
    operands = [sympy2eqtree(arg) for arg in expr.args]
    if merge_numbers and all(isinstance(op, nd.Number) for op in operands):
        return nd.Number(symbol(*operands).eval())
    return symbol(*operands)


def str2sympy(eq:str, variables=['alpha', 'beta', 'gamma', 'I']):
    return sp.parse_expr(eq, local_dict={var:sp.Symbol(var) for var in variables})


def simplify(eq:nd.Symbol):
    try:
        variables = list(set(var.name for var in eq.iter_preorder() if isinstance(var, nd.Variable)))
        return sympy2eqtree(sp.simplify(str2sympy(str(eq), variables)), merge_numbers=True)
    except:
        return eq.copy()


def decompose(eqtrees:nd.Symbol|List[nd.Symbol], 
              route:List[List[nd.Symbol]]=[],
              route2:List[nd.Symbol]=[nd.Empty()]) -> Generator[List[Tuple[List[nd.Symbol], nd.Symbol]], None, None]:
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
    if isinstance(eqtrees, nd.Symbol): eqtrees = [eqtrees]
    if all([eq.n_operands == 0 for eq in eqtrees]): 
        yield list(zip([*route, eqtrees], route2))
    else:
        for idx, eq in enumerate(eqtrees):
            if eq.n_operands == 0: continue
            f = route2[-1].copy()
            cnt = -1
            for n in f.iter_preorder():
                if isinstance(n, nd.Empty): 
                    cnt +=1
                    if cnt == idx: 
                        f = f.replace(n, eq.__class__())
                        break
            else:
                raise Exception('Error')
            yield from decompose([*eqtrees[:idx], *eq.operands, *eqtrees[idx+1:]], 
                                 [*route, eqtrees],
                                 [*route2, f])
