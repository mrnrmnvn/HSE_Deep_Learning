import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import matplotlib.pyplot as plt

writer = SummaryWriter(comment='Segmentation')

torch.manual_seed(2023)

paths = glob("./stage1_train/*")

class DSB2018(Dataset):

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img_path = glob(self.paths[idx] + "/images/*")[0]
        mask_imgs = glob(self.paths[idx] + "/masks/*")
        img = imread(img_path)[:, :, 0:3]  
        img = np.moveaxis(img, -1, 0)
        img = img / 255.0

        masks = [imread(f) / 255.0 for f in mask_imgs]

        final_mask = np.zeros(masks[0].shape)
        for m in masks:
            final_mask = np.logical_or(final_mask, m)
        final_mask = final_mask.astype(np.float32)

        img, final_mask = torch.tensor(img), torch.tensor(final_mask).unsqueeze(0)

        img = F.interpolate(img.unsqueeze(0), (256, 256))
        final_mask = F.interpolate(final_mask.unsqueeze(0), (256, 256))

        return img.type(torch.FloatTensor)[0], final_mask.type(torch.FloatTensor)[0]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        
        
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        
        
        self.conv11 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        
        x1 = F.relu(self.conv2(F.relu(self.conv1(x))))
        x2 = self.pool1(x1)
        
        x3 = F.relu(self.conv4(F.relu(self.conv3(x2))))
        x4 = self.pool2(x3)
        
        
        x5 = F.relu(self.conv6(F.relu(self.conv5(x4))))
        
        
        x6 = self.upconv1(x5)
        x6 = torch.cat([x3, x6], 1)
        x7 = F.relu(self.conv8(F.relu(self.conv7(x6))))
        
        x8 = self.upconv2(x7)
        x8 = torch.cat([x1, x8], 1)
        x9 = F.relu(self.conv10(F.relu(self.conv9(x8))))
        
        
        output = F.sigmoid(self.conv11(x9))
        
        return output


model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_func = nn.BCEWithLogitsLoss()


dsb_data = DSB2018(paths)

train_split, test_split = torch.utils.data.random_split(dsb_data, [500, len(dsb_data)-500])
train_seg_loader = DataLoader(train_split, batch_size=1, shuffle=True)
test_seg_loader = DataLoader(test_split,  batch_size=1)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for x, y in train_seg_loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_func(prediction, y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'unet_model.pt')  # Сохраняем веса модели в файл


# model = UNet()
# model.load_state_dict(torch.load('unet_model.pt'))
# model.eval()  # Переводим модель в режим оценки

