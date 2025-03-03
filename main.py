import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks import PGD, FGSM
import torchsummary
from torchsummary import summary
import shutil
from ptflops import get_model_complexity_info

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

#GTSRB
# Organize test data
data_dir = 'GTSRB/GTSRB_Final_Test_Images/GTSRB'
images_dir = os.path.join(data_dir, 'Final_Test/Images')
test_dir = os.path.join(data_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

with open('GTSRB/GTSRB_Final_Test_GT/GT-final_test.csv') as f:
    image_names = f.readlines()

for text in image_names[1:]:
    classes = int(text.split(';')[-1])
    image_name = text.split(';')[0]

    test_class_dir = os.path.join(test_dir, f"{classes:04d}")
    os.makedirs(test_class_dir, exist_ok=True)
    image_path = os.path.join(images_dir, image_name)

    shutil.copy(image_path, test_class_dir)

batch_size = 15

trainset = torchvision.datasets.ImageFolder(root='GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images',
                                                transform=transforms.Compose([
                                                          transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          ]),
                                               )

testset = torchvision.datasets.ImageFolder(root='GTSRB/GTSRB_Final_Test_Images/GTSRB/test',
                                                transform=transforms.Compose([
                                                          transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          ]),
                                               )

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def plot_images(images, labels, classes, normalize=True):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 20))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

batch = next(iter(train_loader))
classes = trainset.classes
plot_images(batch[0], batch[1], classes)

# Load and modify model
from LNL import LNL_Ti as small
model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)
model = model.cuda()

# Train Locality-iN-Locality
num_epochs = 5
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(num_epochs):
    total_batch = len(trainset) // batch_size

    for i, (batch_images, batch_labels) in enumerate(train_loader):
        X = batch_images.cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f' % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

# Test model
model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))