import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


torch.manual_seed(2023)


class TitanicDataset(Dataset):

    def __init__(self):

        super().__init__()

        self.df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())

        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].mean())

        s_dummies = pd.get_dummies(self.df['Sex'], prefix='Sex')
        self.df = pd.concat([self.df, s_dummies], axis=1)
        self.df['Sex'] = self.df['Sex'].map({'male': 0, 'female': 1})
        
        pclass_dummies = pd.get_dummies(self.df['Pclass'], prefix='Pclass')
        self.df = pd.concat([self.df, pclass_dummies], axis=1)

        self.df['SibSp'] = self.df['SibSp'].fillna(self.df['SibSp'].mode())

        self.df = self.df.dropna()


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        alive = torch.Tensor([1, 0])
        dead = torch.Tensor([0, 1])
        y = alive if row['Survived'] else dead
        x = torch.Tensor([row['Age'], row['Fare'], row['SibSp'], row['Sex'], row['Pclass_1'], row['Pclass_2'], row['Pclass_3']])
        return x, y
    

titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=1, shuffle=True)


class SurvivalPredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x
    

model = SurvivalPredictorPerceptron(input_size=7, hidden_size=50, output_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCELoss()

num_epochs = 100

for epoch in range(num_epochs):
    error = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y.float())
        error += loss.item()

        loss.backward()
        optimizer.step()

    print(error/len(titanic_dataset))
