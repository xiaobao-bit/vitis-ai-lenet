import torch, random
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from config import Config

cfg = Config()

def data_loader(root = cfg.data_dir, batch_size = cfg.batch_size, split = cfg.split, subset_len = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root = root,
        train = True,
        transform = transform,
        download = True
    )

    test_dataset = datasets.CIFAR10(
        root = root,
        train = True,
        transform = transform,
        download = True
    )

    datasets_size = len(train_dataset)
    train_size, calib_size = int(datasets_size * split[0]), int(datasets_size * split[1])
    train_dataset, calib_dataset = random_split(train_dataset, [train_size, calib_size])

    if subset_len:
        assert subset_len <= len(calib_dataset)
        calib_dataset = torch.utils.data.Subset(
            calib_dataset, random.sample(range(0, len(calib_dataset)), subset_len))

    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle = True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size = batch_size,
        shuffle = True
    )
    calib_loader = DataLoader(
        calib_dataset, 
        batch_size = batch_size,
        shuffle = True
    )

    
    return train_loader, calib_loader, test_loader

if __name__ == '__main__':
    train_loader, calib_loader, test_loader = data_loader(subset_len=10)
    for i, (images, labels) in enumerate(calib_loader):
        print(images.shape)
        print(labels.shape)
        break