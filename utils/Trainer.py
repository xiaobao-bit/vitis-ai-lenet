import torch
from tqdm import tqdm
from .Metric import AverageMeter, accuracy, evaluate
from config import Config

cfg = Config()

class Trainer:
    def __init__(self, train_loader, optimizer, device = cfg.device, loss_fn = cfg.loss_fn):
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn

    def train(self, model, epoch=1, model_dir = None):
        model.train()
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)

        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            loss = self.loss_fn(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            progress_bar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%',
                'Acc@5': f'{top5.avg:.2f}%'
            })

if __name__ == '__main__':
    from utils.dataloader import data_loader
    from pt_model.lenet import LeNet 
    import torch.nn as nn

    train_loader, calib_loader, test_loader = data_loader()

    model = LeNet()
    optimizer = cfg.optimizer(model.parameters(), lr=cfg.lr)

    trainer = Trainer(
        train_loader=train_loader,
        optimizer=optimizer
    )

    for epoch in range(cfg.epochs):
        trainer.train(model, epoch)

    if cfg.save_model:
        torch.save(model.state_dict(), './pt_model/lenet.pt')
        print('Model saved')