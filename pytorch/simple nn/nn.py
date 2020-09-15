import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class carDataset():

    def __init__(self):
    # Upload csv files
        file = pd.read_csv('Opel1.csv')
        file2 = pd.read_csv('Opel2.csv')
        file3 = pd.read_csv('Peugeot1.csv')
        file4 = pd.read_csv('Peugeot2.csv')

        # file = file.drop(file.columns[[15, 16]], axis = 1, inplace=True)

        all_files = pd.concat([file, file2, file3, file4], axis=0)

        if (all_files.isnull().sum().sum() != 0):
            all_files.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        x = all_files.iloc[1:18000, 1:14].values
        y = all_files.iloc[1:18000, 16:17].values

        x = np.nan_to_num(x)
        sc = StandardScaler()

        x_train = sc.fit_transform(x)
        y_train = y


        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class carDataset_test():

    def __init__(self):
        # Upload csv files
        file = pd.read_csv('Opel1.csv')
        file2 = pd.read_csv('Opel2.csv')
        file3 = pd.read_csv('Peugeot1.csv')
        file4 = pd.read_csv('Peugeot2.csv')

        # file = file.drop(file.columns[[15, 16]], axis = 1, inplace=True)

        all_files = pd.concat([file, file2, file3, file4], axis=0)

        if (all_files.isnull().sum().sum() != 0):
            all_files.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        x = all_files.iloc[18000:, 1:14].values
        y = all_files.iloc[18000:, 16:17].values
        x = np.nan_to_num(x)

        sc = StandardScaler()

        x_train = sc.fit_transform(x)
        y_train = y

        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)
        print(x_train.shape)
    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 13)
        self.fc2 = nn.Linear(13, 2)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.logsigmoid(x)

net = Net()
print(net)

learning_rate = 0.01

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# Создаем функцию потерь
criterion = nn.CrossEntropyLoss()

epochs = 5

carset = carDataset()
train_loader = torch.utils.data.DataLoader(carset, batch_size = 64, shuffle = True )



# Main Train Loop
for epoch in range(epochs):
   running_loss = 0
   for features, labels in train_loader:
       output = net(features)
       labels = labels.squeeze()
       loss = criterion(output, labels)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       running_loss += loss.item()
   print('Train Epoch: {} with  Loss: {:.6f}'.format(
          epoch, loss.item()))


carset_test = carDataset_test()
test_loader = torch.utils.data.DataLoader(carset_test, batch_size = 64, shuffle = True )

# Test Loop
test_loss = 0
correct = 0
for features, labels in test_loader:
   output = net(features)
   labels = labels.squeeze_()
   test_loss += criterion(output, labels).data.item()
   pred = output.data.max(1)[1]  # получаем индекс максимального значения
   correct += pred.eq(labels.data).sum()


test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

