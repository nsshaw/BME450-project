import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Dataset used: https://www.kaggle.com/datasets/trolukovich/food11-image-dataset?resource=download

#set device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#weird if statement needed to combat some weird error where cpu cant run multiple things idk
if __name__ == '__main__':

    #image transforms
    resize_transform = transforms.Compose([transforms.Resize((64,64)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #batch size
    batch_size = 32

    #dataloaders
    trainset = datasets.ImageFolder(
    root='finalprojectdata/training',
    transform=resize_transform
        )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2, pin_memory=True)

    testset = datasets.ImageFolder(
    root='finalprojectdata/validation',
    transform=resize_transform
        )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2, pin_memory=True)

    #classes to be learned, can change between 10 and 4
    #classes = ('Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup')
    classes = ('Bread',  'Egg',  'Meat', 'Noodles-Pasta')

    #neural network architecture
    class Net(nn.Module):
            def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 channels for RGB, 16 3x3 filters
                    self.pool = nn.MaxPool2d(2, 2) # Max pooling 2x2 filter
                    self.conv2 = nn.Conv2d(16, 8, 3, padding=1)  # 16 inputs, 8 3x3 filters
                    self.dropout = nn.Dropout(0.6) # 0.6 dropout filter
                    self.fc1 = nn.Linear(8 * 16 * 16, 64) # fully connected layer to 64 outputs
                    self.fc2 = nn.Linear(64, 32) # fully connected layer to 32 outputs
                    self.fc3 = nn.Linear(32, 4) # fully connected layer to 10/4 classes

            def forward(self, x):
                    x = self.conv1(x)
                    x = self.pool(F.relu(x))
                    x = self.conv2(x)
                    x = self.pool(F.relu(x))
                    x = torch.flatten(x, 1) # flatten all dimensions except batch
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
             
                    return x


    #initialize network and move to GPU
    net = Net()
    net.to(device)

    import torch.optim as optim

    #setup the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.01)

    num_epochs = 50

    #lists to keep track of stats
    train_loss_over_time = []
    test_loss_over_time = []
    train_accuracy_over_time = []
    test_accuracy_over_time = []

    #training and testing loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        #Training loop
        net.train()  # set the model to training mode
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss_over_time.append(running_loss / len(trainloader))
        train_accuracy_over_time.append(100 * correct / total)

        #Testing loop
        net.eval()  # set the model to evaluation mode
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss_over_time.append(test_loss / len(testloader))
        test_accuracy_over_time.append(100 * correct / total)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss_over_time[-1]:.3f}, Train Acc: {train_accuracy_over_time[-1]:.2f}%, Test Loss: {test_loss_over_time[-1]:.3f}, Test Acc: {test_accuracy_over_time[-1]:.2f}%')

    #Plotting the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_loss_over_time, label='Training Loss')
    plt.plot(range(num_epochs), test_loss_over_time, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracy_over_time, label='Training Accuracy')
    plt.plot(range(num_epochs), test_accuracy_over_time, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epoch')
    plt.legend()

    plt.show()