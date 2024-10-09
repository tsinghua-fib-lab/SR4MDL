import os
import torch
import logging
import numpy as np
from sr4mdl.utils import set_seed, set_proctitle, set_signal, init_logger, AutoGPU, get_args, get_fig
from sr4mdl.utils.metrics import RMSE_score, R2_score, kendall_rank_score, spearman_rank_score, pearson_score, AUC_score, NDCG_score

# Get args
args = get_args(save_dir='./results/test')

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


if __name__ == '__main__':
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
    from sr4mdl.model.mdlformer import MDLformer
    mdlformer = MDLformer(args, dataset.get_token_list())
    logger.info(mdlformer)

    # Recovery Checkpoint (if specified)
    state_dict = torch.load(args.load_model, map_location=args.device, weights_only=False)
    state_dict['xy_encoder'] = {k.removeprefix('module.'): v for k, v in state_dict['xy_encoder'].items()}
    mdlformer.load(state_dict['xy_encoder'], state_dict['xy_token_list'], strict=True)
    logger.info(f'Loaded checkpoint from {args.load_model}')
    mdlformer.eval()

    true, pred = [], []
    for idx, (data, prefix, used_vars, length) in enumerate(loader):
        pred.extend(mdlformer.predict(data).cpu().tolist())
        true.extend(length.tolist())
        if not idx % 10: logger.info(f'Processed {idx} ({len(true)})')
        if len(true) >= 1024: break
    logger.info(str(pred))
    logger.info(str(true))
    logger.note(f'RMSE: {RMSE_score(true, pred):.4f}')
    logger.note(f'R2: {R2_score(true, pred):.4f}')
    logger.note(f'Kendall: {kendall_rank_score(true, pred):.4f}')
    logger.note(f'Spearman: {spearman_rank_score(true, pred):.4f}')
    logger.note(f'Pearson: {pearson_score(true, pred):.4f}')
    logger.note(f'AUC: {AUC_score(true, pred):.4f}')
    logger.note(f'NDCG@5: {NDCG_score(true, pred, k=5):.4f}')

    fi, fig, axes = get_fig(1, 1, AW=5, AH=5)
    axes[0].scatter(true, pred, s=1)
    axes[0].set_xlabel('True')
    axes[0].set_ylabel('Pred')
    fig.savefig(os.path.join(args.save_dir, 'plot.png'))
