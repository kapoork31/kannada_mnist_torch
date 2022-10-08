import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        """
        initialize net class attributes 
            including conv layers,
            fully connected layers/dense layers
            dropout
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        define CNN network architecture
        Quite a general network following CNN development principles
            of using relu, max pooling, dropout etc.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(epoch, model, train_loader, device, optimizer, train_loss):
    """
    function to train the model using gradient descent and back propogation
    """
    model.train()
    # function used to train the model
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        # start of each batch zero our gradient
        optimizer.zero_grad()
        data = data.permute(0, 3, 1, 2).float()
        output = model(data)
        loss = criterion(output, target)
        # calculate our loss for each batch
        loss.backward()
        # back propogate the error/loss through the network
        optimizer.step()
        
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                optimizer.state_dict()['param_groups'][0]['lr'],
                loss.data))
    train_loss.append(loss.data.cpu().numpy())

def evaluate(model, test_loader, alternate_test_loader, device, test_loss, test_accuracy, alternate_test_loss, alternate_test_accuracy):
    """
    function to test model on test loss data
    and on the alternate test data

    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            data = data.permute(0, 3, 1, 2).float()
            output = model(data)

            loss += criterion(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()
    
    loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    test_loss.append(loss.cpu().numpy())
    test_accuracy.append(accuracy)

    print('test loss: {:.4f}, test accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * accuracy))     

    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in alternate_test_loader:
            data = data.to(device)
            target = target.to(device)
            # permutation required to change order of dimensions in data variable
            # the input channel should be in the second dimension in pytorch
            data = data.permute(0, 3, 1, 2).float()
            output = model(data)

            loss += criterion(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()
    
    loss /= len(alternate_test_loader.dataset)
    accuracy = correct / len(alternate_test_loader.dataset)
    
    alternate_test_loss.append(loss.cpu().numpy())
    alternate_test_accuracy.append(accuracy)
    
    print('alternate test loss: {:.4f}, alternate test accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(alternate_test_loader.dataset),
        100. * accuracy))     


def criterion(input, target, size_average=True):
    """
    Categorical cross-entropy with logits input and one-hot target
    """
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l