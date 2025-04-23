# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class LeNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(LeNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #LeNet::input_0(LeNet::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #LeNet::LeNet/Conv2d[conv1]/ret.3(LeNet::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #LeNet::LeNet/ret.5(LeNet::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #LeNet::LeNet/MaxPool2d[pool]/337(LeNet::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #LeNet::LeNet/Conv2d[conv2]/ret.7(LeNet::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #LeNet::LeNet/ret.9(LeNet::nndct_relu_5)
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #LeNet::LeNet/MaxPool2d[pool]/375(LeNet::nndct_maxpool_6)
        self.module_7 = py_nndct.nn.Module('nndct_flatten') #LeNet::LeNet/ret.11(LeNet::nndct_flatten_7)
        self.module_8 = py_nndct.nn.Linear(in_features=400, out_features=120, bias=True) #LeNet::LeNet/Linear[fc1]/ret.13(LeNet::nndct_dense_8)
        self.module_9 = py_nndct.nn.ReLU(inplace=False) #LeNet::LeNet/ret.15(LeNet::nndct_relu_9)
        self.module_10 = py_nndct.nn.Linear(in_features=120, out_features=84, bias=True) #LeNet::LeNet/Linear[fc2]/ret.17(LeNet::nndct_dense_10)
        self.module_11 = py_nndct.nn.ReLU(inplace=False) #LeNet::LeNet/ret.19(LeNet::nndct_relu_11)
        self.module_12 = py_nndct.nn.Linear(in_features=84, out_features=10, bias=True) #LeNet::LeNet/Linear[fc3]/ret(LeNet::nndct_dense_12)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(input=output_module_0, start_dim=1, end_dim=-1)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        return output_module_0
