import torch
import torch.nn as nn
from models.networks.base_network import BaseNetwork

# Defines the SuperResolution discriminator with the specified arguments.
class t1Discriminator(BaseNetwork):  #axial
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.norm = nn.LayerNorm

        nf = opt.ndf
        input_nc = 1

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(opt.ndf, opt.ndf*2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf*2),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf*2, opt.ndf*4, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf*4),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf*4, opt.ndf*8, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf*8),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 8, opt.ndf * 16, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 16),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf*16, 1, kernel_size=4, stride=1, padding=1)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):    #len = 6
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
class t2Discriminator(BaseNetwork):  #axial
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.norm = nn.LayerNorm

        nf = opt.ndf
        input_nc = 1

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(opt.ndf, opt.ndf * 2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf * 2),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 2, opt.ndf * 4, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf * 4),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 4, opt.ndf * 8, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 8),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 8, opt.ndf * 16, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 16),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 16, 1, kernel_size=4, stride=1, padding=1)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
class flairDiscriminator(BaseNetwork):  #axial
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.norm = nn.LayerNorm

        nf = opt.ndf
        input_nc = 1

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(opt.ndf, opt.ndf * 2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf * 2),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 2, opt.ndf * 4, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf * 4),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 4, opt.ndf * 8, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 8),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 8, opt.ndf * 16, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 16),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 16, 1, kernel_size=4, stride=1, padding=1)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
class t1ceDiscriminator(BaseNetwork):  #axial
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.norm = nn.LayerNorm

        nf = opt.ndf
        input_nc = 1

        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(opt.ndf, opt.ndf * 2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf * 2),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 2, opt.ndf * 4, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(nf * 4),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 4, opt.ndf * 8, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 8),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 8, opt.ndf * 16, kernel_size=4, stride=1, padding=1),
                      nn.BatchNorm2d(nf * 16),
                      nn.LeakyReLU(0.2, True)
                      ]]
        sequence += [[nn.Conv2d(opt.ndf * 16, 1, kernel_size=4, stride=1, padding=1)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

