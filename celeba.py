import torch
import torchvision
from torchvision import transforms
from tqdm import *
tqdm.pos = 0

image_size = (28, 28)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

celeb_a_train = torchvision.datasets.CelebA('D:\Fairlearn\CelebA', split="train", transform=transform)
print(celeb_a_train)

data_loader = torch.utils.data.DataLoader(celeb_a_train,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=4
                                          )

print(celeb_a_train.attr[:, 20])

# for a, b in celeb_a_train:
#     print(b.numpy())
