import torch
from pytorch_nndct.apis import Inspector as VaiInspector
from config import Config

cfg = Config()

class VAI_Inspector:
    def __init__(self, model_class, model_name):
        self.model_class = model_class
        self.model_name = model_name

    def inspect(self):
        inspector = VaiInspector(cfg.target)
        model = self.model_class()
        dummy_input = torch.randn(*cfg.input_img_size).to(cfg.device)
        inspector.inspect(
            model,
            (dummy_input,),
            device=cfg.device,
            output_dir=f"{cfg.inspector_output_dir}/{self.model_name}"
        )

if __name__ == '__main__':
    from pt_model.lenet import LeNet
    inspector = VAI_Inspector(LeNet, "LeNet")
    inspector.inspect()