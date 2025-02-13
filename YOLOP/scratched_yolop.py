import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, num_samples=100, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像和标签
        img = np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)
        bbox = np.random.rand(4)  # [x, y, w, h]
        mask = np.random.randint(0, 2, (self.img_size, self.img_size)).astype(np.float32)

        img = self.transform(img)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return img, bbox, mask

# 定义一个多任务模型
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # 共享的卷积层
        self.shared_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 目标检测分支
        self.bbox_head = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 输出边界框 [x, y, w, h]
        )

        # 分割分支
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # 输出分割掩码
        )

    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_layers(x)

        # 目标检测分支
        bbox_features = shared_features.view(shared_features.size(0), -1)
        bbox_output = self.bbox_head(bbox_features)

        # 分割分支
        mask_output = self.mask_head(shared_features)

        return bbox_output, mask_output

# 训练函数
def train_model(model, dataloader, criterion_bbox, criterion_mask, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, bboxes, masks in dataloader:
            imgs, bboxes, masks = imgs.cuda(), bboxes.cuda(), masks.unsqueeze(1).cuda()

            # 前向传播
            bbox_preds, mask_preds = model(imgs)

            # 计算损失
            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            loss_mask = criterion_mask(mask_preds, masks)
            loss = loss_bbox + loss_mask

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# 主函数
def main():
    # 数据集和数据加载器
    dataset = SimpleDataset(num_samples=1000, img_size=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型、损失函数和优化器
    model = MultiTaskModel().cuda()
    criterion_bbox = nn.MSELoss()  # 边界框回归损失
    criterion_mask = nn.BCELoss()  # 分割掩码损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, dataloader, criterion_bbox, criterion_mask, optimizer, num_epochs=10)

if __name__ == "__main__":
    main()