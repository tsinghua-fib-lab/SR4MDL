import torch
import numpy as np
from nd2py.utils import init_logger, AutoGPU, AttrDict
from sr4mdl.env import Tokenizer
from sr4mdl.search import MCTS4MDL
from sr4mdl.model import MDLformer

init_logger('sr4mdl', exp_name='regressor', info_level='note')

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
    uniform_sample_number=200, # number of samples
    device='auto', # device to run the model, can be 'auto', 'cuda', 'cpu'
    train_eval_split=0.25, # ratio of the training samples to be used for evaluation
    load_model='/path/to/weights/checkpoint.pth',
)

if args.device == 'auto':
    args.device = AutoGPU().choice_gpu(memory_MB=1486, interval=15)

tokenizer = Tokenizer(-100, 100, 4, args.max_var)
state_dict = torch.load(args.load_model, map_location=args.device, weights_only=False)
mdlformer = MDLformer(args, state_dict['xy_token_list'])
mdlformer.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
mdlformer.eval()

eval_kwargs = {
    'scale_x': False,
    'scale_y': False,
    'max_train_samples': int(args.uniform_sample_number / args.train_eval_split)
}

est = MCTS4MDL(
    tokenizer=tokenizer, 
    model=mdlformer,
    n_iter=1_000_000,
    binary=['Mul', 'Div', 'Add', 'Sub'],
    unary=['Sqrt', 'Cos', 'Sin', 'Pow2', 'Pow3', 'Exp', 'Log', 'Inv', 'Neg'],
    # unary=['Sqrt', 'Cos', 'Sin', 'Pow2', 'Pow3', 'Exp', 'Log', 'Inv', 'Neg', 'Arcsin', 'Arccos', 'Cot', 'Log', 'Tanh'],
    # unary=['Cos', 'Sin', 'Pow2', 'Neg'],
    leaf=[1, 2, np.pi],
    log_per_sec=180, # print log info per 3min
    const_range=None,
    keep_vars=True,
    normalize_y=False,
    normalize_all=False,
    remove_abnormal=True,
    train_eval_split=args.train_eval_split
)

def complexity(est):
    return len(est.eqtree)

def model(est, X=None):
    return est.eqtree.to_str()
