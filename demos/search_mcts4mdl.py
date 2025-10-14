import os
import json
import yaml
import time
import torch
import signal
import logging
import traceback
import numpy as np
import nd2py as nd2
import pandas as pd
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from nd2py.utils import seed_all, init_logger, AutoGPU, AttrDict
from sr4mdl.utils import parse_parser, RMSE_score, R2_score
from sr4mdl.search import MCTS4MDL
from sr4mdl.model import MDLformer
from sr4mdl.env import sympy2eqtree, str2sympy, Tokenizer


parser = ArgumentParser()
parser.add_argument('-f', '--function', type=str, default='f=x1+x2*sin(x3)', help='`f=...\' or `Feynman_xxx\'')
parser.add_argument('-n', '--name', type=str, default=None)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='auto')
parser.add_argument('--sample_num', type=int, default=200)
parser.add_argument('--c', type=float, default=1.41)
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--max_var', type=int, default=10)
parser.add_argument('--load_model', type=str, default='./weights/checkpoint.pth')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--keep_vars', action='store_true')
parser.add_argument('--normalize_y', action='store_true')
parser.add_argument('--normalize_all', action='store_true')
parser.add_argument('--remove_abnormal', action='store_true')
parser.add_argument('--use_old_model', action='store_true')
parser.add_argument('--cheat', action='store_true')


args = parse_parser(parser, save_dir='./results/search/')

init_logger('sr4mdl', args.name, os.path.join(args.save_dir, 'info.log'))
logger = logging.getLogger('sr4mdl.search')
logger.info(args)
seed_all(args.seed)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
setproctitle(f'{args.name}@YuZihan')

if args.device == 'auto':
    args.device = AutoGPU().choice_gpu(memory_MB=1486, interval=15)
args.function = args.function.replace(' ', '')


def search():
    if '=' in args.function:
        f = sympy2eqtree(str2sympy(args.function.split('=', 1)[1]))
        binary = list(set(op.__class__ for op in f.iter_preorder() if op.n_operands == 2))
        unary = list(set(op.__class__ for op in f.iter_preorder() if op.n_operands == 1))
        leaf = list(set(op for op in f.iter_preorder() if isinstance(op, nd2.Number)))
        variables = list(set(op.name for op in f.iter_preorder() if isinstance(op, nd2.Variable)))
        X = {var: np.random.uniform(-5, 5, (args.sample_num,)) for var in variables}
        y = f.eval(X)
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
    else:
        import pmlb
        logger.info(f'fetching {args.function} from PMLB...')
        os.makedirs('./data/cache', exist_ok=True)
        df = pmlb.fetch_data(args.function, local_cache_dir='./data/cache/')
        if df.shape[0] > args.sample_num: 
            df = df.sample(args.sample_num)
        else: 
            args.sample_num = df.shape[0]
        logger.info(f'Done, df.shape = {df.shape}')
        X = {col:df[col].values for col in df.columns}
        y = X.pop('target')
        binary = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]
        unary = [nd2.Sqrt, nd2.Cos, nd2.Sin, nd2.Pow2, nd2.Pow3, nd2.Exp, nd2.Inv, nd2.Neg, nd2.Arcsin, nd2.Arccos, nd2.Cot, nd2.Log, nd2.Tanh]
        leaf = [nd2.Number(1), nd2.Number(2), nd2.Number(np.pi)]
        log = {
            'target function': args.function,
            'binary operators': [op.__name__ for op in binary],
            'unary operators': [op.__name__ for op in unary],
            'leaf': [op.to_str(number_format=".2f") for op in leaf],
            'variables': list(X.keys()),
        }
        try:
            metadata = yaml.load(open(f'./data/pmlb/datasets/{args.function}/metadata.yaml', 'r'), Loader=yaml.Loader)['description']
            metadata = [l.strip() for l in metadata.split('\n')]
            target, eq = metadata[metadata.index('')+1].split(' = ', 1)
            eq = sympy2eqtree(str2sympy(eq))
            log['target function'] = log['target function'] + ' ({} = {})'.format(target, eq.to_str(number_format=".2f"))
            if args.cheat:
                binary = list(set(op.__class__ for op in eq.iter_preorder() if op.n_operands == 2))
                unary = list(set(op.__class__ for op in eq.iter_preorder() if op.n_operands == 1))
                leaf = list(set(op for op in eq.iter_preorder() if isinstance(op, nd2.Number)))
                log['binary operators'] = [op.__name__ for op in binary]
                log['unary operators'] = [op.__name__ for op in unary]
                log['leaf'] = [op.to_str(number_format=".2f") for op in leaf]
        except Exception as e:
            logger.warning(e)
    logger.note('\n'.join(f'{k}: {v if not isinstance(v, list) else "[" + ", ".join(v) + "]"}' for k, v in log.items()))

    tokenizer = Tokenizer(-100, 100, 4, args.max_var)
    state_dict = torch.load(args.load_model)
    model_args = AttrDict(dropout=0.1, d_model=512, d_input=64, d_output=512, n_TE_layers=8, max_len=50, max_param=5, max_var=args.max_var, uniform_sample_number=args.sample_num,device=args.device, use_SENet=True, use_old_model=args.use_old_model)
    model = MDLformer(model_args, state_dict['xy_token_list'])
    model.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
    model.eval()
    est = MCTS4MDL(
        tokenizer=tokenizer,
        model=model,
        n_iter=args.n_iter,
        keep_vars=args.keep_vars,
        normalize_y=args.normalize_y,
        normalize_all=args.normalize_all,
        remove_abnormal=args.remove_abnormal,
        binary=binary,
        unary=unary,
        leaf=leaf,
        log_per_sec=5,
        save_path=os.path.join(args.save_dir, 'records.json'),
    )

    try:
        est.fit(X, y, use_tqdm=False)
        logger.info('Finished')
    except KeyboardInterrupt:
        logger.note('Interrupted')
    except Exception:
        logger.error(traceback.format_exc())

    y_pred = est.predict(X)
    rmse = RMSE_score(y, y_pred)
    r2 = R2_score(y, y_pred)
    logger.note(f'Result = {est.eqtree}, RMSE = {rmse:.4f}, R2 = {r2:.4f}')

    result = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'host': gethostname(),
        'name': args.name,
        'load_model': args.load_model,
        'success': str(rmse < 1e-6),
        'n_iter': len(est.records),
        'duration': est.records[-1]['time'],
        'model': 'MCTS4MDL',
        'exp': args.function,
        'result': str(est.eqtree),
        'rmse': rmse,
        'r2': r2,
        'sample_num': args.sample_num,
        'seed': args.seed,
    }
    json.dump(result, open(os.path.join(args.save_dir, 'result.json'), 'w'), indent=4)

    # aggregate results to aggregate.csv
    save_path = './results/aggregate.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write('\t'.join([
                'success','name','exp','n_iter','duration','seed','rmse','r2',
                'result','target','date','host','load_model','model','sample_num'
            ]) + '\n')
    with open(save_path, 'a') as f:
        keys = open(save_path, 'r').readline().split('\t')
        f.write(','.join(str(result.get(k, '')) for k in keys) + '\n')


if __name__ == '__main__':
    search()
