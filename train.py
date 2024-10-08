import os
import json
import logging
import traceback
import numpy as np
import torch.nn as nn
from sr4mdl.utils import set_seed, set_proctitle, set_signal, init_logger, AutoGPU, get_args

# Get args
args = get_args(save_dir='./results')

# Init
init_logger(exp_name=args.name, log_file=os.path.join(args.save_dir, 'info.log'), quiet=args.quiet)
logger = logging.getLogger('my.main')
logger.info(args)
set_signal()
set_seed(args.seed)
set_proctitle(f'{args.name}@YuZihan')

# Refine Args
if args.num_workers: args.save_equations = int(np.ceil(args.save_equations / args.num_workers))
assert not (args.load_model and args.continue_from), 'Cannot use --load_model and --continue_from at the same time'
if ',' in args.device:  # e.g., "cuda:0,1,2,3"
    args.device_ids = [int(i) for i in args.device.removeprefix('cuda:').split(',')]
    args.device = f'cuda:{args.device_ids[0]}'
if args.device == 'auto':
    memory_MB = min(1150 + args.batch_size * 154, 80000)
    args.device = AutoGPU().choice_gpu(memory_MB=memory_MB)


def train():
    # Load dataset
    if args.dataset == 'default':
        from sr4mdl.generator import Num2EqDataset
        dataset = Num2EqDataset(args, beyond_token=True)
    elif args.dataset == 'hard': 
        from sr4mdl.generator import Num2EqDatasetHard
        dataset = Num2EqDatasetHard(args, beyond_token=True)
    elif args.dataset == 'pure':
        from sr4mdl.generator import Num2EqDatasetPure
        dataset = Num2EqDatasetPure(args, beyond_token=True)
    elif args.dataset == 'keep':
        from sr4mdl.generator import Num2EqDatasetKeep
        dataset = Num2EqDatasetKeep(args, beyond_token=True)
    elif args.dataset == 'load':
        from sr4mdl.generator import Num2EqDatasetLoad
        from sr4mdl.env import sympy2eqtree, str2sympy
        with open('./data/load_eqtrees.txt') as f:
            eqtrees = [sympy2eqtree(str2sympy(eq)) for eq in f.readlines()]
        dataset = Num2EqDatasetLoad(eqtrees, args, beyond_token=True)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    loader = dataset.get_dataloader(batch_size=args.batch_size, num_workers=args.num_workers)

    # Load model
    from sr4mdl.model.mdlformer import MDLformer, FormulaEncoder, Trainer
    mdlformer = MDLformer(args, dataset.get_token_list())
    eq_encoder = FormulaEncoder(args, dataset.get_token_list(all=True))
    if 'device_ids' in args:
        mdlformer = nn.DataParallel(mdlformer, device_ids=args.device_ids)
        eq_encoder = nn.DataParallel(eq_encoder, device_ids=args.device_ids)
    trainer = Trainer(args, mdlformer, eq_encoder)
    logger.info(mdlformer)
    logger.info(eq_encoder)

    # Recovery Checkpoint (if specified)
    if args.continue_from is not None:
        result_dir = 'results'
        trainer.load(os.path.join(result_dir, args.continue_from, 'checkpoint.pth'), abs_path=True, model_only=False)
        with open(os.path.join(result_dir, args.continue_from, 'records.json'), 'r') as f:
            trainer.records = [json.loads(line) for line in f.readlines()]
            trainer.records = trainer.records[:trainer.n_step]
        if os.path.join(result_dir, args.continue_from) != args.save_dir:
            with open(os.path.join(args.save_dir, 'records.json'), 'w') as f:
                for record in trainer.records:
                    f.write(json.dumps(record) + '\n')
            trainer.plot(f'plot.png')
        logger.note(f'Continue from ./{result_dir}/{args.continue_from} at step {trainer.n_step}')
    elif args.load_model is not None:
        trainer.load(args.load_model, abs_path=True, model_only=True)
        logger.note(f'Load model from {args.load_model}')

    # Training
    try:
        for idx, (data, prefix, used_vars, length) in enumerate(loader):
            log = trainer.step(data, prefix, used_vars, length, detail=not (trainer.n_step+1) % 10)
            if (not trainer.n_step % 10) or (idx < 10): 
                logger.info(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
            if trainer.n_step >= args.tot_steps: break
        logger.note('Training finished:' + (' | '.join(f'{k}:{v}' for k, v in log.items())))
    except KeyboardInterrupt:
        logger.warning(f'Interrupted at step {trainer.n_step}')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        if trainer.n_step >= 100:
            trainer.save('checkpoint.pth')
            trainer.plot('plot.png')


if __name__ == '__main__':
    train()
