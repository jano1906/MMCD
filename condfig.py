import argparse


def setting_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_log_dir', type=str, default='data/output/log',
                        help='Path to log file.')

    parser.add_argument('--out_checkpoint_dir', type=str, default='data/output/MMCD',
                        help='Path to checkpoint file.')

    parser.add_argument('--save_top_k', type=int, default=10,
                        help='save_top_k for train.')

    parser.add_argument('--gpus', type=int, default=0,
                        help='gpus for train.')

    parser.add_argument('--n_max_epochs', type=int, default=500,
                        help='max_epochs for train.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_sizes for train.')

    parser.add_argument('--n_timestep', type=int, default=1000,
                        help='n_timestep for diffusion model.')

    parser.add_argument('--beta_schedule', type=str, default='linear',
                        help='beta_schedule for diffusion model.')

    parser.add_argument('--beta_start', type=float, default=1.e-7,
                        help='beta_start for diffusion model.')

    parser.add_argument('--beta_end', type=float, default=2.e-2,
                        help='beta_end for diffusion model.')

    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature for diffusion model.')

    parser.add_argument('--learning_rate_struct', type=float, default=5e-3,
                        help='learning_rate_struct for diffusion model.')

    parser.add_argument('--learning_rate_seq', type=float, default=5e-3,
                        help='learning_rate_seq for diffusion model.')

    parser.add_argument('--learning_rate_cont', type=float, default=5e-3,
                        help='learning_rate_cont for diffusion model.')

    return parser.parse_args()
