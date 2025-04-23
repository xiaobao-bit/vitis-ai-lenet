# vitis-ai-lenet
LeNet as example to use Vitis AI for xmodel

## Env
WSL2 + i9

Vitis-AI3.0 + pytorch-cpu

## To run the code
1. Enter vitis ai docker
2. Go to xmodel_generator dir
3. Execute the following sequentially:
   ```
   python -m utils.dataloader
   python -m utils.Trainer
   python -m vai_flow.Inspector
   python -m vai_flow.Quantizer
   sudo chmod +x vai_flow/Compiler.sh
   sh vai_flow/Compiler.sh
   ```
