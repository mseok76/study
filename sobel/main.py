from PIL import Image
import torchvision.transforms as transforms
from torchvision.traforms import ToPILImage
from model import Sobel_filter
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--name",type = str, default = 'image')
arg = parser.parse_args()

def main():
    img_path = './dataset/' + arg.name
    img = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    Sobel = Sobel_filter()
    sobel_image = Sobel(img_tensor)
    sobel_image = ToPILImage(sobel_image.unsqueeze())
    save_path = './result/'+time.time()+'.jpg'
    sobel_image.save(save_path)





