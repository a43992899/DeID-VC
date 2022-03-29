import os
import argparse
from solver_encoder import Solver, Solver_enhanced
from data_loader import get_loader
from torch.backends import cudnn
from hparams import DATA_PATH


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    # cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop)
    
    solver = Solver_enhanced(vcc_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd_reconst', type=float, default=10, help='weight for content code reconstruction loss')
    parser.add_argument('--lambda_cd_convert', type=float, default=10, help='weight for content code convertion loss')
    parser.add_argument('--lambda_emb_convert', type=float, default=0.001, help='weight for speaker embedding convertion loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default=DATA_PATH+'./spmel')
    parser.add_argument('--resume', type=str, default=None) # "/home/yrb/code/ID-DEID/data/model/enhanced_freq32/autovc_epoch_98_loss_0.0350.ckpt"
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)