import numbers
import logging
import numpy as np
import sympy as sp
from typing import List, Dict
from functools import reduce
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", message="RuntimeWarning: invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="RuntimeWarning: invalid value encountered in log")
warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered in exp")
warnings.filterwarnings("ignore", message="RuntimeWarning: invalid value encountered in power")
warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered in multiply")

# ignore RuntimeWarning: invalid value encountered in add/sub/mul/divide
np.seterr(invalid='ignore')
# ignore RuntimeWarning: overflow encountered in exp
np.seterr(over='ignore')
# ignore RuntimeWarning: divide by zero encountered in power
np.seterr(divide='ignore')


logger = logging.getLogger('my.symbols')


class MetaSymbol(type):
    def __repr__(cls):
        return cls.__name__

class Symbol(metaclass=MetaSymbol):
    n_operands = None
    def __init__(self, *operands):
        self.parent = None
        self.child_idx = None

        operands = list(operands) if len(operands) else [Empty() for _ in range(self.n_operands)]
        for idx, op in enumerate(operands):
            if type(op) in [float, int]:
                operands[idx] = Number(op)

        self.operands = operands
        assert len(self.operands) == self.n_operands

        for idx, operand in enumerate(self.operands):
            operand.parent = self
            operand.child_idx = idx

    @classmethod
    def __repr__(cls):
        return cls.__name__
        
    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def __len__(self):
        return 1 + sum(len(operand) for operand in self.operands)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        for op1, op2 in zip(self.operands, other.operands):
            if op1 != op2: return False
        return True
    
    def __hash__(self):
        return hash(self.to_str())

    @classmethod
    def create_instance(cls, *operands):
        self = cls(*operands)
        if all(isinstance(x, Number) for x in operands): self = Number(self.eval())
        return self

    def to_str(self, **kwargs):
        """
        Args:
        - raw:bool=False, whether to return the raw format
        - number_format:str='', can be '0.2f'
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        - latex:bool=False, whether to return the latex format
        """
        name = self.__class__.__name__
        if not kwargs.get('raw'): name = name.lower()
        return f'{name}({", ".join(x.to_str(**kwargs) for x in self.operands)})'

    def to_tree(self, **kwargs):
        """
        Args:
        - number_format:str='', can be '0.2f'
        """
        if self.n_operands == 0: return f'{self.to_str(**kwargs)}'
        name = f'{self.__class__.__name__}'
        children = [operand.to_tree(**kwargs) for operand in self.operands]
        for idx, child in enumerate(children):
            children[idx] = ('├ ' if idx < len(children)-1 else '└ ') + child.replace('\n', '\n' + ('┆ ' if idx < len(children)-1 else '  '))
        return name + '\n' + '\n'.join(children)

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def preorder(self):
        yield self
        for operand in self.operands:
            yield from operand.preorder()
    
    def postorder(self):
        for operand in self.operands:
            yield from operand.postorder()
        yield self
    
    def __add__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Add(self, other)

    def __radd__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Add(other, self)

    def __sub__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Sub(self, other)

    def __rsub__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Sub(other, self)

    def __mul__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Mul(other, self)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Div(self, other)
    
    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Div(other, self)

    def __pow__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Pow(self, other)
    
    def __rpow__(self, other):
        if isinstance(other, numbers.Number): other = Number(other)
        return Pow(other, self)
    
    def __neg__(self):
        return Neg(self)

    def is_constant(self, **kwargs):
        return all([op.is_constant(**kwargs) for op in self.operands])
    
    def replace(self, symbol:'Symbol'):
        if self.parent is None: 
            logger.warning(f'"{self}" has no parent in replacement for "{symbol}"!')
            return symbol
        self.parent.operands[self.child_idx] = symbol
        symbol.parent = self.parent
        symbol.child_idx = self.child_idx
        self.parent = None
        self.child_idx = None
        return symbol
    
    def copy(self):
        return self.__class__(*[op.copy() for op in self.operands])
    
    def fit(self, X:Dict[str,np.ndarray], y:np.ndarray, maxiter=30, method='BFGS'):
        # 替身机制反而更慢了
        sutando = self
        parameters = [op for op in sutando.preorder() if isinstance(op, Number) and op.fitable]
        if len(parameters) == 0: return self  # 没有 fitable Number
        def set_params(params):
            for p, param in enumerate(parameters): param.value = params[p]
        def loss(params):
            set_params(params)
            return np.mean((y - sutando.eval(**X)) ** 2)
        x0 = np.array([param.value for param in parameters])
        res = minimize(loss, x0, method=method, options={'maxiter': maxiter})
        set_params(res.x)
        return self

    def create_sutando(self, *args, **kwargs) -> 'Symbol':
        """ 
        使用启发式的方法创建一个替身，与 self 共享 fitable Number
        替身的形式更加简洁，能够更快速地被 fit，且 fit 过程中 self 的 fitable Number 也会被更新
        """
        if self.n_operands == 0:
            if isinstance(self, Number) and self.fitable: return self  # 需要拟合的量
            else: return Number(self.eval(*args, **kwargs), fitable=False)  # 不需要拟合的量
        sutando_operands = [op.create_sutando(*args, **kwargs) for op in self.operands]
        if all(isinstance(op, Number) and not op.fitable for op in sutando_operands):  # 没有 fitable Number 的子公式
            return Number(self.__class__(*sutando_operands).eval(*args, **kwargs), fitable=False)
        return self.__class__(*sutando_operands)  # 有 fitable Number 且难以继续简化的子公式

    def get_depth(self, max_depth=100, root_depth=0):
        if root_depth >= max_depth: return max_depth
        if self.n_operands == 0: return root_depth
        return max(operand.get_depth(max_depth, root_depth+1) for operand in self.operands)


class Empty(Symbol):
    n_operands = 0
    def to_str(self, **kwargs):
        if kwargs.get('raw', False): return 'Empty()'
        if kwargs.get('latex', False): return r'□'
        return '?'

    def eval(self, *args, **kwargs):
        raise ValueError('Incomplete Equation Tree')

    def is_constant(self, **kwargs):
        return False
    
    def __len__(self):
        return 0


class Number(Symbol):
    n_operands = 0
    def __init__(self, value, fitable=True):
        super().__init__()
        if isinstance(value, Number): value = value.value
        self.value = value
        self.fitable = fitable

    def to_str(self, **kwargs):
        fmt = kwargs.get('number_format', '')
        content = f'{self.value:{fmt}}'
        return content if self.fitable else f'Constant({content})'
        
    def __eq__(self, other: int|float) -> bool:
        if not isinstance(other, (Number, numbers.Number)): return False
        return self.value == (other.value if isinstance(other, Number) else other)

    def eval(self, *args, **kwargs):
        return self.value

    def is_constant(self, **kwargs):
        return True
    
    def copy(self):
        return self.__class__(self.value, self.fitable)
    
    def __hash__(self):
        return hash(self.value)


class Variable(Symbol):
    n_operands = 0
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def to_str(self, **kwargs):
        if kwargs.get('raw', False): return f'Variable("{self.name}")'
        if kwargs.get('latex', False): return f'{self.name[0]}_{{{self.name[1:]}}}'
        return self.name 
    
    def eval(self, *args, **kwargs):
        return kwargs[self.name]
        # return eval(self.name, globals(), kwargs)

    def is_constant(self, **kwargs):
        return self.name in kwargs
    
    def copy(self):
        return self.__class__(self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Variable): return False
        return self.name == other.name


class Add(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        return f'{x1} + {x2}'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) + self.operands[1].eval(*args, **kwargs)
   
    @classmethod
    def create_instance(self, *operands):
        p, num = 0, Number(0.0)
        operands = list(operands)
        while p < len(operands):
            if operands[p].__class__ == Number: 
                num.value += operands.pop(p).value
            else: p += 1
        add = [operand for operand in operands if operand.__class__ != Neg]
        sub = [operand.operands[0] for operand in operands if operand.__class__ == Neg]
        if num.value != 0.0: add.append(num)
        if len(sub) == 0: 
            return reduce(lambda x, y: Add(x, y), add)
        elif len(add) == 0:
            return Neg(reduce(lambda x, y: Add(x, y), sub))
        else: 
            return Sub(reduce(lambda x, y: Add(x, y), add), reduce(lambda x, y: Add(x, y), sub))


class Sub(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if self.operands[1].__class__ in [Add, Sub]: x2 = f'({x2})'
        return f'{x1} - {x2}'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) - self.operands[1].eval(*args, **kwargs)


class Mul(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub]: x1 = f'({x1})'
        if self.operands[1].__class__ in [Add, Sub]: x2 = f'({x2})'
        if self.operands[1].__class__ in [Add, Sub, Variable] and kwargs.get('omit_mul_sign', False): return f'{x1}{x2}'
        return f'{x1} * {x2}' if not kwargs.get('latex', False) else rf'{x1} \times {x2}'
    
    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) * self.operands[1].eval(*args, **kwargs)
   
    @classmethod
    def create_instance(self, *operands):
        operands = list(operands)
        
        p, num = 0, Number(1.0)
        while p < len(operands):
            if operands[p].__class__ == Number: 
                num.value *= operands.pop(p).value
            else: p += 1
        
        for p, op in enumerate(operands):
            if isinstance(op, Neg):
                num.value *= -1
                operands[p] = op.operands[0]

        numer = [operand for operand in operands if operand.__class__ != Inv]
        if num.value != 1.0 and num.value != -1.0: numer.insert(0, num)
        denom = [operand.operands[0] for operand in operands if operand.__class__ == Inv]
        if len(denom) == 0: 
            res = reduce(lambda x, y: Mul(x, y), numer)
        elif len(numer) == 0:
            res = Inv(reduce(lambda x, y: Mul(x, y), denom))
        else: 
            res = Div(reduce(lambda x, y: Mul(x, y), numer), reduce(lambda x, y: Mul(x, y), denom))
        if num.value == -1.0: return Neg(res)
        return res

class Div(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\frac{{{x1}}}{{{x2}}}'
        if self.operands[0].__class__ in [Add, Sub]: x1 = f'({x1})'
        if self.operands[1].__class__ in [Add, Sub, Mul, Div, Inv]: x2 = f'({x2})'
        return f'{x1} / {x2}'
    
    def eval(self, *args, **kwargs):
        return np.divide(self.operands[0].eval(*args, **kwargs), self.operands[1].eval(*args, **kwargs))


class Pow(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub, Mul, Div, Pow, Neg, Inv, Pow2, Pow3]: x1 = f'({x1})'
        elif self.operands[0].__class__ == Number and self.operands[0].value < 0: x1 = f'({x1})'
        assert x1[0] != '-', x1 # FOR DEBUG
        if kwargs.get('latex', False): return rf'{x1}^{{{x2}}}'
        if self.operands[1].__class__ in [Add, Sub, Mul, Div, Inv]: x2 = f'({x2})'
        return f'{x1} ** {x2}'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) ** self.operands[1].eval(*args, **kwargs)
    
    @classmethod
    def create_instance(self, *operands):
        if operands[1] == 0.5: return Sqrt(operands[0])
        if operands[1] == 2: return Pow2(operands[0])
        if operands[1] == 3: return Pow3(operands[0])
        if operands[1] == -1: return Inv(operands[0])
        if operands[1] == -0.5: return Inv(Sqrt(operands[0]))
        if operands[1] == -2: return Inv(Pow2(operands[0]))
        if operands[1] == -3: return Inv(Pow3(operands[0]))
        return Pow(*operands)


class Cat(Symbol):
    n_operands = 2
    def __init__(self, *operands):
        super().__init__(*operands)
        assert all([isinstance(operand, Number) for operand in operands])

    def __str__(self):
        return f'{self.operands[0]}{self.operands[1]}'

    def eval(self, *args, **kwargs):
        return int(str(self))


class Max(Symbol):
    n_operands = 2
    def eval(self, *args, **kwargs):
        x1 = self.operands[0].eval(*args, **kwargs)
        x2 = self.operands[1].eval(*args, **kwargs)
        return np.maximum(x1, x2)

    def create_instance(self, *operands):
        return reduce(lambda x, y: Max(x, y), operands)


class Min(Symbol):
    n_operands = 2
    def eval(self, *args, **kwargs):
        x1 = self.operands[0].eval(*args, **kwargs)
        x2 = self.operands[1].eval(*args, **kwargs)
        return np.minimum(x1, x2)
    
    def create_instance(self, *operands):
        return reduce(lambda x, y: Min(x, y), operands)


class Sin(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.sin(self.operands[0].eval(*args, **kwargs))

class Cos(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.cos(self.operands[0].eval(*args, **kwargs))

class Tan(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.tan(self.operands[0].eval(*args, **kwargs))

class Log(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.log(self.operands[0].eval(*args, **kwargs))

class Exp(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.exp(self.operands[0].eval(*args, **kwargs))

class Arcsin(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.arcsin(self.operands[0].eval(*args, **kwargs))

class Arccos(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.arccos(self.operands[0].eval(*args, **kwargs))

class Arctan(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.arctan(self.operands[0].eval(*args, **kwargs))

class Sqrt(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.sqrt(self.operands[0].eval(*args, **kwargs))    
    
class Abs(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return abs(self.operands[0].eval(*args, **kwargs))

class Neg(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        # if self.operands[0].__class__ in 
        return f'-({x})'
    
    def eval(self, *args, **kwargs):
        return -self.operands[0].eval(*args, **kwargs)

class Inv(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\frac{{1}}{{{x}}}'
        return f'1 / ({x})'

    def eval(self, *args, **kwargs):
        return np.divide(1, self.operands[0].eval(*args, **kwargs))

class Pow2(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub, Mul, Div, Pow, Neg, Inv, Pow2, Pow3]: x = f'({x})'
        elif self.operands[0].__class__ == Number and self.operands[0].value < 0: x = f'({x})'
        assert x[0] != '-', x # FOR DEBUG
        return f'{x} ** 2' if not kwargs.get('latex', False) else f'{x}^2'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) ** 2

class Pow3(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub, Mul, Div, Pow, Neg, Inv, Pow2, Pow3]: x = f'({x})'
        elif self.operands[0].__class__ == Number and self.operands[0].value < 0: x = f'({x})'
        assert x[0] != '-', x # FOR DEBUG
        return f'{x} ** 3' if not kwargs.get('latex', False) else f'{x}^3'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) ** 3

class Tanh(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.tanh(self.operands[0].eval(*args, **kwargs))

class Sigmoid(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return 1 / (1 + np.exp(-self.operands[0].eval(*args, **kwargs)))

class Regular(Symbol):
    n_operands = 2
    def eval(self, *args, **kwargs):
        x1 = self.operands[0].eval(*args, **kwargs)
        x2 = self.operands[1].eval(*args, **kwargs)
        return x1 ** x2 / (1 + x1 ** x2)


class Cot(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return 1 / np.tan(self.operands[0].eval(*args, **kwargs))
