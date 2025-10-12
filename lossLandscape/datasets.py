import torch
import torchvision
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, models, transforms

def load_cifar10(train_batch_size=128, test_batch_size=128):
    
    train_T = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]) 
    
    test_T = transforms.Compose([
        transforms.ToTensor(),
    ]) 
    
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, transform=train_T, download=True) 
    testset  = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_T) 
    
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(trainset.classes)
    print("Number of classes:", num_classes)
    
    return trainset, testset, trainloader, testloader, num_classes


def load_cifar100(train_batch_size=128, test_batch_size=128):
    
    train_T = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]) 
    
    test_T = transforms.Compose([
        transforms.ToTensor(),
    ]) 
    
    trainset = torchvision.datasets.CIFAR100(root='data', train=True, transform=train_T, download=True) 
    testset  = torchvision.datasets.CIFAR100(root='data', train=False, transform=test_T) 
    
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(trainset.classes)
    print("Number of classes:", num_classes)

    return trainset, testset, trainloader, testloader, num_classes

def load_caltech256(train_batch_size=128, test_batch_size=128):

    torch.manual_seed(0)
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]) 

    dataset = torchvision.datasets.Caltech256(root="./data", download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = 257 # The extra one is a clutter class
    print("Number of classes:", num_classes)
    
    return trainset, testset, trainloader, testloader, num_classes

def load_flowers(train_batch_size=128, test_batch_size=128):
    
    train_T = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]) 
    
    test_T = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]) 
    
    trainset = torchvision.datasets.Flowers102(root='data', split='train', transform=train_T, download=True) 
    valset = torchvision.datasets.Flowers102(root='data', split='val', transform=test_T) 
    testset  = torchvision.datasets.Flowers102(root='data', split='test', transform=test_T) 

    trainval_concatset = ConcatDataset([trainset, valset])
    
    trainloader = DataLoader(trainval_concatset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = 102
    print("Number of classes:", num_classes)
    
    return trainset, testset, trainloader, testloader, num_classes

def load_dogs(train_batch_size=128, test_batch_size=128):

    ### The code for the Stanford Dogs dataset is inspired by: https://github.com/zrsmithson/Stanford-dogs/tree/master
    ### Please place the file stanford_dogs_data.py in the same folder as here.
    ### We thank the authors of the code for the well-working implementation of Stanford Dogs in PyTorch

    from stanford_dogs_data import dogs
    
    train_T = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]) 
    
    test_T = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        
    ]) 
    
    trainset = dogs(root='data', train=True, cropped=False, transform=train_T, download=True)
    testset = dogs(root='data', train=False, cropped=False, transform=test_T, download=True)
    
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader  = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = 120
    print("Number of classes:", num_classes)
    
    return trainset, testset, trainloader, testloader, num_classes