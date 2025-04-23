import torch
import torch.nn as nn
import torch.optim as optim

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataset parameters
        self.input_img_size = [1, 3, 32, 32]
        self.batch_size = 32
        self.data_dir = "./data"
        self.split = [0.7, 0.3]

        # training parameters
        self.save_model = True
        self.optimizer = optim.Adam
        self.epochs = 3
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = 0.001

        # inspector:
        self.target = "DPUCAHX8L_ISA0_SP"
        self.inspector_output_dir = "./inspect_res"

        # quantization:
        self.quant_batch_size = 16
        self.calib_subset = 200
        
