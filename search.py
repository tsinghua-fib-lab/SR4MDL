import os
import yaml
import pmlb
import time
import torch
import signal
import logging
import traceback
import numpy as np
from argparse import ArgumentParser
from setproctitle import setproctitle
from sr4mdl.utils import set_seed, init_logger, AutoGPU, AttrDict
from sr4mdl.env import sympy2eqtree, str2sympy
from sr4mdl.env.symbols import Add, Sub, Mul, Div, Pow, Sqrt, Cos, Sin, Pow2, Pow3, Exp, Inv, Neg, Arcsin, Arccos, Cot, Log, Tanh, Number, Variable, Symbol


def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


parser = ArgumentParser()
parser.add_argument('-m', '--model', choices=['GP', 'MCTS', 'SSGP', 'SSMCTS', 'SSGPwoNN', 'SSMCTSwoNN'], default='SSGPwoNN')
parser.add_argument('-f', '--function', type=str, default='f=x0+x1*sin(x2)+x0*x1*x2')
parser.add_argument('-n', '--name', type=str, default=None)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-d', '--device', type=str, default='auto')
parser.add_argument('-N', '--sample_num', type=int, default=300)
parser.add_argument('--c', type=float, default=1.41)
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--load_model', type=str, default='./log/finetune800/checkpoint.pth')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--keep_vars', action='store_true')
parser.add_argument('--normalize_y', action='store_true')
parser.add_argument('--normalize_all', action='store_true')
parser.add_argument('--remove_abnormal', action='store_true')
parser.add_argument('--use_old_model', action='store_true')
parser.add_argument('--cheat', action='store_true')
args, unknown = parser.parse_known_args()
args = AttrDict(vars(args))
args.seed = args.seed or np.random.randint(1, 32768)
args.name = args.name or f'{args.model}-{time.strftime("%Y%m%d%H%M%S")}'
args.save_dir = f'./log/{args.name}'

args.max_var = 10

set_seed(args.seed)
setproctitle(f'{args.name}@YuZihan')
os.makedirs(args.save_dir, exist_ok=True)
init_logger(args.name, os.path.join(args.save_dir, 'info.log'), quiet=args.quiet)
logger = logging.getLogger('my.main')
if unknown: logger.warning(f'Unknown args: {unknown}')
logger.info(args)


def search():
    os.makedirs('./result/', exist_ok=True)
    if not os.path.exists('./result/search-results.csv'):
        with open('./result/search-results.csv', 'w') as f:
            f.write('\t'.join([
                'date', 'host', 'name', 'load_model',
                'success', 'n_iter', 'duration',
                'model', 'exp', 'result', 'mse', 'r2', 
                'sample_num', 'seed'
            ]) + '\n')

    if args.device == 'auto' and args.model in ['SSGP', 'SSMCTS']:
        memory_MB = 1486
        args.device = AutoGPU().choice_gpu(memory_MB=memory_MB, interval=15)

    if args.function.startswith('f='):
        f = sympy2eqtree(str2sympy(args.function.split('=', 1)[1]))
        binary = list(set(op.__class__ for op in f.preorder() if op.n_operands == 2))
        unary = list(set(op.__class__ for op in f.preorder() if op.n_operands == 1))
        leaf = list(set(op for op in f.preorder() if isinstance(op, Number)))
        variables = list(set(op.name for op in f.preorder() if isinstance(op, Variable)))
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
        X = {var: np.random.uniform(-5, 5, (args.sample_num,)) for var in variables}
        y = f.eval(**X)
    else:
        logger.info(f'fetching {args.function} from PMLB...')
        os.makedirs('./data/PMLB_cache/', exist_ok=True)
        df = pmlb.fetch_data(args.function, local_cache_dir=f'./data/PMLB_cache/{args.function}')
        if df.shape[0] > args.sample_num: df = df.sample(args.sample_num)
        logger.info(f'Done, shape = {df.shape}')
        X = {col:df[col].values for col in df.columns}
        y = X.pop('target')
        args.sample_num = len(y)
        binary = [Mul, Div, Add, Sub] # Pow
        unary = [Sqrt, Cos, Sin, Pow2, Pow3, Exp, Inv, Neg, Arcsin, Arccos, Cot, Log, Tanh]
        leaf = [Number(1), Number(2), Number(np.pi)]
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
        # try:
        eqtrees = yaml.load(open('./data/feynman_strogatz_eqs.yaml', 'r'), Loader=yaml.Loader)
        target, eq = eqtrees[args.function].split(' = ', 1)
        eq = sympy2eqtree(str2sympy(eq))
        log['target function'] = log['target function'] + ' ({} = {})'.format(target, eq.to_str(number_format=".2f"))
        if args.cheat:
            binary = list(set(op.__class__ for op in eq.preorder() if op.n_operands == 2))
            unary = list(set(op.__class__ for op in eq.preorder() if op.n_operands == 1))
            leaf = list(set(op for op in eq.preorder() if isinstance(op, Number)))
            log['binary operators'] = [op.__name__ for op in binary]
            log['unary operators'] = [op.__name__ for op in unary]
            log['leaf'] = [op.to_str(number_format=".2f") for op in leaf]
        # except Exception as e:
        #     logger.error(e)
    logger.note('\n'.join(f'{k}: {v if not isinstance(v, list) else "[" + ", ".join(v) + "]"}' for k, v in log.items()))

    kwargs = {
        'binary': binary,
        'unary': unary,
        'leaf': leaf,
        # 'log_per_iter': 1,
        'log_per_sec': 5,
        'const_range': None,
        'save_path': os.path.join(args.save_dir, 'records.json'),
    }

    from sr4mdl.search import MCTS4MDL
    from sr4mdl.model.mdlformer import MDLformer
    from sr4mdl.env.tokenizer import Tokenizer
    tokenizer = Tokenizer(-100, 100, 4, args.max_var)
    state_dict = torch.load(args.load_model, map_location=args.device, weights_only=False)
    model_args = AttrDict(dropout=0.1, d_model=512, d_input=64, d_output=512, n_TE_layers=8, max_len=50, max_param=5, max_var=args.max_var, uniform_sample_number=args.sample_num,device=args.device, use_SENet=True, use_old_model=args.use_old_model)
    model = MDLformer(model_args, state_dict['xy_token_list'])
    model.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
    model.eval()
    kwargs['keep_vars'] = args.keep_vars
    kwargs['normalize_y'] = args.normalize_y
    kwargs['normalize_all'] = args.normalize_all
    kwargs['remove_abnormal'] = args.remove_abnormal
    est = MCTS4MDL(tokenizer=tokenizer, model=model, **kwargs)

    try:
        est.fit(X, y, n_iter=args.n_iter, use_tqdm=False)
    except KeyboardInterrupt:
        logger.note('Interrupted')
    except Exception:
        logger.error(traceback.format_exc())

    y_pred = est.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - mse / np.var(y)
    logger.note(f'Result = {est.eqtree}, MSE = {mse:.4f}, R2 = {r2:.4f}')

    with open('./result/search-results.csv', 'a') as f:
        f.write('\t'.join(map(str, [
            # 'date', 'host', 'name', 'load_model',
            # 'success', 'n_iter', 'duration',
            # 'model', 'exp', 'result', 'mse', 'r2', 
            # 'sample_num', 'seed'
            time.strftime('%Y-%m-%d %H:%M:%S'), os.uname().nodename, args.name, args.load_model,
            mse < 1e-6, len(est.records), est.records[-1]['time'], 
            args.model, args.function, est.eqtree, mse, r2, 
            args.sample_num, args.seed, 
        ])) + '\n')


if __name__ == '__main__':
    search()
