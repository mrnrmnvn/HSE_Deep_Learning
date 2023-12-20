from torchvision import datasets, transforms
import torch
from matplotlib import pyplot
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='CIFAR100 hw')

train_indices = torch.arange(7500)
train_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=transforms.ToTensor(), train=True)
train_cifar_dataset = data_utils.Subset(train_cifar_dataset, train_indices)

test_indices = torch.arange(2000)
test_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=transforms.ToTensor(), train=False)
test_cifar_dataset = data_utils.Subset(test_cifar_dataset, test_indices)

train_cifar_dataloader = DataLoader(dataset=train_cifar_dataset, batch_size=8, shuffle=True)
test_cifar_dataloader = DataLoader(dataset=test_cifar_dataset, batch_size=8, shuffle=True)

sample = train_cifar_dataset[20]
print(sample[0].size())

class CifarPredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x

model = CifarPredictorPerceptron(input_size=3*32*32, hidden_size=180, output_size=100)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 50

for epoch in range(num_epochs):
    error1 = 0
    correct_guess = 0
    # total_images = 0
    for x, y in train_cifar_dataloader:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)

        loss = loss_fn(prediction, y)
        error1 += loss.item()

        predicted_indices = torch.argmax(prediction, dim=1)
        correct_guess += (predicted_indices == y).sum().item()
        # total_images += y.size(0)

        loss.backward()
        optimizer.step()


    # train_accuracy = correct_guess / total_images
    train_accuracy = correct_guess / len(train_cifar_dataset)
    writer.add_scalar('Train Accuracy', train_accuracy, epoch)
    writer.add_scalar('Train Loss', error1/len(train_cifar_dataset), epoch)


    test_correct_guess = 0
    # total_images = 0
    error2 = 0
    for x, y in test_cifar_dataloader:
        model.eval()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error2 += loss.item()

        predicted_indices = torch.argmax(prediction, dim=1)
        test_correct_guess += (predicted_indices == y).sum().item()
        # total_images += y.size(0)

    # test_accuracy = correct_guess / total_images
    test_accuracy = test_correct_guess / len(test_cifar_dataset)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)
    writer.add_scalar('Test Loss', error2/len(test_cifar_dataset), epoch)

print('done')



