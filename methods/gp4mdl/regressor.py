import torch
import nd2py as nd
from nd2py.utils import init_logger, AutoGPU, AttrDict
from sr4mdl.env import Tokenizer
from sr4mdl.search import GP4MDL
from sr4mdl.model import MDLformer

init_logger("sr4mdl", exp_name="regressor")

args = AttrDict(
    ## Fixed parameters
    dropout=0.0, 
    d_model=512, 
    d_input=64, 
    d_output=512, 
    n_TE_layers=8, 
    max_var=10,
    ## Adjustable parameters
    max_len=50, # maximum length of the equation enabled to search
    max_param=5, # maximum number of parameters in the equation
    uniform_sample_number=2000000, # number of samples
    device='auto', # device to run the model, can be 'auto', 'cuda', 'cpu'
    train_eval_split=0.25, # ratio of the training samples to be used for evaluation(when liner regression)
    load_model='./weights/checkpoint.pth',
)


if args.device == 'auto':
    args.device = AutoGPU().choice_gpu(memory_MB=1486, interval=15)

tokenizer = Tokenizer(-100, 100, 4, args.max_var)
state_dict = torch.load(args.load_model, map_location=args.device, weights_only=False)
mdlformer = MDLformer(args, state_dict['xy_token_list'])
mdlformer.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
mdlformer.eval()

# 注意：SRBench 的 experiments/evaluate_model.py:101 有误，应该将 X_train = X_train.loc[sample_idx] 改成 X_train = X_train.iloc[sample_idx]
eval_kwargs = {
    'scale_x': False,
    'scale_y': False,
    'max_train_samples': int(args.uniform_sample_number / args.train_eval_split)
}

# 由于 GP4MDL 的初始化依赖于 variables，但 SRBench 只在调用 fit 时才会传入 variables，
# 因此必须在外面包装一层 GP4MDLRegressor 以缓存输入的参数，直到被调用 fit 时才初始化一个 GP4MDL 用于真正的 fit
class GP4MDLRegressor(GP4MDL):
    def fit(self, X, y, timeout=np.inf):
        variables = [nd.Variable(col, nettype='scalar') for col in X.keys()]
        self.real_est = GP4MDL(tokenizer=tokenizer, 
                               model=mdlformer,
                               variables=variables, 
                               binary=self.binary, 
                               unary=self.unary, 
                               n_iter=self.n_iter,
                               log_per_iter=self.log_per_iter,
                               keep_vars=self.keep_vars,
                               normalize_y=self.normalize_y,
                               normalize_all=self.normalize_all,
                               remove_abnormal=self.remove_abnormal,
                               eps=self.eps,
                               train_eval_split=self.train_eval_split,
                               )
        return self.real_est.fit(X, y, timeout=timeout)

    def predict(self, X):
        return self.real_est.predict(X)


    def refit_val_best(self, X, y): 
        return self.real_est.refit_val_best(X, y)

est = GP4MDLRegressor(
    tokenizer=tokenizer, 
    model=mdlformer,
    variables=None, # <= 在这里时还不知道变量什么，要等到调用 GP4MDLRegressor.fit 时才知道
    binary=[nd.Add, nd.Sub, nd.Mul, nd.Div],
    unary=[nd.Sqrt, nd.Sin, nd.Cos, nd.Neg, nd.Inv, nd.Log, nd.Exp, nd.Pow2, nd.Pow3],
    n_iter=200, #这里做了修改，注意
    log_per_iter=10,
    keep_vars=False,
    normalize_y=True,
    normalize_all=True,
    remove_abnormal=False,
    train_eval_split=args.train_eval_split
)


def complexity(est):
    return len(est.real_est.val_best_eqtree)


def model(est, X=None):
    return str(est.real_est.val_best_eqtree)
