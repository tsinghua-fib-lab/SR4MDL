import numpy as np
import nd2py as nd

class Tokenizer:
    def __init__(self, min_exponent, max_exponent, n_mantissa, max_var):
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.n_mantissa = n_mantissa
        self.max_var = max_var

        self.MAX_VALUE = (10 - 10 / 10**self.n_mantissa) * 10**self.max_exponent
        self.EPS_VALUE = 10 / 10**self.n_mantissa * 10**self.min_exponent

        self.token_list = self.get_token_list(all=True)
        self.token2index = {token: i for i, token in enumerate(self.token_list)}
        self.vectorize = np.vectorize(self.token2index.get)

    def get_token_list(self, all=False):
        token_list = ['+', '-']
        token_list.extend('N' + str(x).zfill(self.n_mantissa) for x in range(10**self.n_mantissa))
        token_list.extend(f'E{x:+03d}' for x in range(self.min_exponent, self.max_exponent+1))
        if all:
            token_list.extend([f'x_{i+1}' for i in range(self.max_var)])
            token_list.extend(['add', 'sub', 'mul', 'div', 'pow', 
                               'sin', 'cos', 'tan', 'log', 'exp', 
                               'arcsin', 'arccos', 'arctan', 'sqrt', 
                               'abs', 'neg', 'inv', 'pow2', 'pow3',
                               'cot', 'sec', 'csc', 'sinh', 'cosh', 'tanh'])
        return token_list

    def float2token(self, data:np.ndarray, nan_to_pad=False):
        """
        Args:
            data: (*) of float
        Returns:
            token: (*, 3) of string
        """
        if not isinstance(data, np.ndarray): data = np.array(data)
        shape = data.shape
        sgn, man, exp = self.float_embed(data.ravel())
        sgn = np.where(sgn, '-', '+')
        man = np.array(['N' + str(x).zfill(self.n_mantissa) for x in man.flatten()]).reshape(man.shape)
        exp = np.array([f'E{x:+03d}' for x in exp.flatten()]).reshape(exp.shape)
        token = np.stack([sgn, man, exp], axis=-1).reshape(*shape, 3)
        if nan_to_pad: token[np.isnan(data), :] = 'PAD'
        return token

    def float2index(self, data:np.ndarray, nan_to_pad=False):
        """
        Args:
            data: (*) of float
        Returns:
            index: (*, 3) of int
        """
        if not isinstance(data, np.ndarray): data = np.array(data)
        shape = data.shape
        sgn, man, exp = self.float_embed(data.ravel())
        sgn = 1 + sgn  # skip <PAD>
        man = 1 + 2 + man # skip <PAD>, +, -
        exp = 1 + 2 + 10**self.n_mantissa + exp - self.min_exponent # skip <PAD>, +, -, N000~N999
        index = np.stack([sgn, man, exp], axis=-1).reshape(*shape, 3)
        if nan_to_pad: index[np.isnan(data), :] = 0
        return index

    def eqtree2token(self, eqtree:np.ndarray):
        token = []
        for node in eqtree.iter_preorder():
            if isinstance(node, nd.Number):
                token.extend(self.float2token(node.value).tolist())
            elif isinstance(node, nd.Variable):
                token.append(node.name)
            else:
                token.append(node.__class__.__name__.lower())
        return token

    def eqtree2index(self, eqtree:np.ndarray):
        token = self.eqtree2token(eqtree)
        return self.vectorize(token)

    def float_embed(self, data:np.ndarray):
        """
        (+|-)(N000~N999)(E-10~E10)
            1.234e5 -> (+, N123, E5)
            -1.234e-5 -> (-, N123, E-5)
            (+-)inf -> (+|-, N999, E10)
            (+-)eps -> (+|-, N000, E-10)
            zero -> (+, N000, E-10)
            nan -> (+, N000, E10)
        """
        nan = np.isnan(data)
        data[nan] = 0.0

        sign = (data < 0).astype(int)
        data = np.abs(data)

        data[data == 0] = np.finfo(float).eps

        exponent = np.floor(np.log10(data)).clip(self.min_exponent, self.max_exponent)
        exponent[nan] = self.max_exponent
        data /= 10**exponent
        exponent = exponent.astype(int)

        mantissa = np.round(data * 10**(self.n_mantissa-1)).clip(0, 10**self.n_mantissa-1).astype(int)
        return sign, mantissa, exponent
