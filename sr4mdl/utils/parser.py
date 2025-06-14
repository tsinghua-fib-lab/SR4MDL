import os
import time
import random
import logging
from argparse import ArgumentParser
from nd2py.utils import AttrDict

logger = logging.getLogger(__name__)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['default', 'hard', 'pure', 'keep', 'load'], default='default')
    parser.add_argument('-n', '--name', type=str, default=None, help='YYYYMMDD_HHMMSS by default')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--tot_steps', type=int, default=300_000)
    parser.add_argument('--device', type=str, default='auto', help='cpu | cuda | cuda:0 | auto | cuda:0,1,2')
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_input', type=int, default=64)
    parser.add_argument('--d_output', type=int, default=512)
    parser.add_argument('--n_TE_layers', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--max_param', type=int, default=5)
    parser.add_argument('--max_var', type=int, default=10)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--save_equations', type=int, default=0, help='保存 N 个生成的公式')
    parser.add_argument('--AMP', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--uniform_sample_number', type=int, default=400, help='若为 0, 采样 100~500 个样本点')
    parser.add_argument('--pred_loss_weight', type=float, default=1.0)
    parser.add_argument('--clip_loss_weight', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--no-simplify', action='store_false', dest='simplify') # simplify 默认为 True
    parser.add_argument('--simplify', action='store_true')

    parser.add_argument('--no-normalize_loss', action='store_false', dest='normalize_loss') # normalize_loss 默认为 True
    parser.add_argument('--normalize_loss', action='store_true')

    parser.add_argument('--no-normalize_X', action='store_false', dest='normalize_X') # normalize_X 默认为 True
    parser.add_argument('--normalize_X', action='store_true')
    parser.add_argument('--no-normalize_y', action='store_false', dest='normalize_y') # normalize_y 默认为 True
    parser.add_argument('--normalize_y', action='store_true')

    return parser


def parse_parser(parser: ArgumentParser, save_dir='./results'):
    args, unknown = parser.parse_known_args()

    if unknown: logger.warning(f'Unknown args: {unknown}')
    args.name = args.name or f'{time.strftime("%Y%m%d_%H%M%S")}'
    args.seed = args.seed or random.randint(1, 32768)

    args.save_dir = os.path.join(save_dir, args.name)
    if os.path.exists(args.save_dir) and getattr(args, 'continue_from', None) != args.name:
        args.name = args.name + '-' + time.strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(save_dir, args.name)
    os.makedirs(args.save_dir, exist_ok=True)

    args = AttrDict(vars(args))
    return args


def get_args(save_dir='./results') -> AttrDict:
    parser = get_parser()
    args = parse_parser(parser, save_dir=save_dir)
    return args
