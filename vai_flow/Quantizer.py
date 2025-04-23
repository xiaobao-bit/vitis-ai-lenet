import torch
from tqdm import tqdm
from pytorch_nndct.apis import torch_quantizer, Inspector
from config import Config
from utils.dataloader import data_loader  
from utils.Metric import evaluate

cfg = Config()

class VAI_Quantizer:
    def __init__(self, model_class, model_path, model_name):
        self.model_class = model_class
        self.model_path = model_path
        self.model_name = model_name
        self.model = self.model_class()
        self.quant_model = None

    def quantizer_calib(self):
        device = cfg.device
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

        batch_size = cfg.quant_batch_size
        input = torch.randn([batch_size, *cfg.input_img_size[1:]])

        quantizer = torch_quantizer('calib', self.model, (input,), device=device)
        self.quant_model = quantizer.quant_model

        self.model.eval()
        _, calib_loader, _ = data_loader(subset_len=cfg.calib_subset)
        for iteraction, (images, labels) in tqdm(enumerate(calib_loader), total=len(calib_loader)):
            outputs = self.quant_model(images)

        quantizer.export_quant_config()

        print("Calibration config exported successfully.")

    def quantizer_exam(self):
        if self.quant_model is None:
            print("Quant model is not calibrated. Run quantizer_calib() first.")
            return

        _, _, test_loader = data_loader(batch_size=cfg.quant_batch_size)
        loss_fn = cfg.loss_fn.to(cfg.device)

        acc1, acc5, loss = evaluate(self.model, test_loader, loss_fn)
        print(f'Float model - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%, Loss: {loss:.4f}')

        acc1_q, acc5_q, loss_q = evaluate(self.quant_model, test_loader, loss_fn)
        print(f'Quant model - Top-1: {acc1_q:.2f}%, Top-5: {acc5_q:.2f}%, Loss: {loss_q:.4f}')


    def quantizer_export(self):
        input = torch.randn(cfg.input_img_size)
        quantizer = torch_quantizer('test', self.model, (input), device=cfg.device)
        quant_model = quantizer.quant_model
        
        _, quant_loader, _ = data_loader(subset_len=1)
        image, label = next(iter(quant_loader))
        output = quant_model(image)
        
        quantizer.export_xmodel(deploy_check=False)
        quantizer.export_onnx_model()


if __name__ == '__main__':
    from pt_model.lenet import LeNet
    quant = VAI_Quantizer(LeNet, 'pt_model/lenet.pt', 'LeNet')
    quant.quantizer_calib()
    quant.quantizer_exam()
    quant.quantizer_export()