import torch
import torchvision
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
#import enumerate

from model import vgg_net

parser = argparse.ArgumentParser()
parser.add_argument("--activation",type = str, default = 'ReLU')
parser.add_argument("--lr",type = float,default=0.0001)

arg = parser.parse_args()

print(f"\n\nActivation : {arg.activation}\tLearning rate : {arg.lr}")

DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(DEVICE+"\n")
def main():
    #dataload
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        
        ])

    root_dir = '.mnist/dataset/'
    trainset = torchvision.datasets.MNIST(root = root_dir + 'train/', download = True, train = True, transform=transform)
    validset = torchvision.datasets.MNIST(root = root_dir + 'valid/', download = True, train = False, transform=transform)

    num_train = len(trainset)
    num_valid = len(validset)

    print(f"Train data \t{num_train}\nValid data \t{num_valid}")

    vgg = vgg_net(arg.activation).to(DEVICE)
    
    train_loader = DataLoader(trainset, batch_size = 64, shuffle = True)
    valid_loader = DataLoader(validset, batch_size = 64, shuffle = True)

    optimizer = torch.optim.Adam(vgg.parameters(),lr = arg.lr)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        vgg.train()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = sample
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = vgg(inputs)
            _, predicted = torch.max(outputs, 1)
            train_correct = torch.sum(predicted == labels).item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += train_correct

        print(f"Train Loss : {train_acc:.4f}\tTrain Acc : {train_acc:.4f}")
        
        with torch.no_grad():
            vgg.eval()
            for i,sample in enumerate(valid_loader):
                optimizer.zero_grad()
                inputs, labels = sample
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = vgg(inputs)
                _, predicted = torch.max(outputs, 1)
                valid_correct = torch.sum(predicted == labels).item()
                loss = criterion(outputs,labels)
                valid_loss += loss.item()
                valid_acc += valid_correct

            print(f"Valid Loss : {valid_acc:.4f}\tValid Acc : {valid_acc:.4f}")


if __name__ == '__main__':
    main()
