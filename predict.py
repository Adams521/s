import numpy as np
from PIL import Image

from siamese import Siamese

if __name__ == "__main__":
    model = Siamese()
        
    # while True:
        # image_1 = input('Input image_1 filename:')
    image_1 = '/home/hozon/bug/Siamese-pytorch/split_data/Sim130/Sim130_001.png'
    # try:
    image_1 = Image.open(image_1)
    # except:
    #     print('Image_1 Open Error! Try again!')
    #     continue

    # image_2 = input('Input image_2 filename:')
    image_2 = '/home/hozon/bug/Siamese-pytorch/split_data/Sim130/Sim130_002.png'
    # try:
    image_2 = Image.open(image_2)
    # except:
    #     print('Image_2 Open Error! Try again!')
    #     # continue
    probability = model.detect_image(image_1,image_2)
    print(probability)
