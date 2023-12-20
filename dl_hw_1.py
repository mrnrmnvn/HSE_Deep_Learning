# Сейчас будем оценивать, насколько сложной будет тренировка, в зависимости от необходимой:
# 1) силы
# 2) выносливости
# 3) гибкости

import torch


stretching = torch.tensor([[0.1, 0.2, 0.7]])
dancing = torch.tensor([[0.1, 0.3, 0.4]])
functional = torch.tensor([[0.7, 0.7, 0.5]])
boxing = torch.tensor([[0.6, 0.5, 0.3]])
cycling = torch.tensor([[0.3, 0.8, 0.2]])


# Создаем датасет из растяжки, танцев, функциональной тренировки, бокса и велотренировки, присваиваем им уровень сложности
dataset = [
    (stretching, torch.tensor([[0.3]])),
    (dancing, torch.tensor([[0.1]])),
    (functional, torch.tensor([[0.9]])),
    (boxing, torch.tensor([[0.7]])),
    (cycling, torch.tensor([[0.4]]))
]


# Присваиваем веса руками
# weights = torch.tensor([[0.75, 0.85, 0.25]], requires_grad=True)
# bias = torch.tensor([[0.1]], requires_grad=True)


# Автоматическое решение вопроса весов
torch.manual_seed(2023)
weights = torch.rand((1, 3), requires_grad=True)
bias = torch.rand((1, 1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=[weights, bias], lr=1e-6)
num_epochs = 10


def predict_difficulty(obj: torch.Tensor) -> torch.Tensor :
    return obj @ weights.T + bias


def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)


for i in range(num_epochs):
    for x, y in dataset:
        optimizer.zero_grad()
        difficulty_score = predict_difficulty(x)
        loss = calc_loss(difficulty_score, y)
        loss.backward()
        print(loss)

        optimizer.step()
