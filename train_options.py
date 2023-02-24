"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=30000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=30000, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adamw')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lrG', type=float, default=0.0004, help='initial learning rate for adam')
        parser.add_argument('--lrD', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of generator iterations per discriminator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_pix', type=float, default=100, help='weight for pixel loss')
        parser.add_argument('--lambda_hf', type=float, default=10, help='weight for hf loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD_t1', type=str, default='t1', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_t2', type=str, default='t2', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_flair', type=str, default='flair', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_t1ce', type=str, default='t1ce', help='(n_layers|multiscale|image)')

        parser.add_argument('--n_layers_D', type=int, default='4')
        self.isTrain = True
        return parser
