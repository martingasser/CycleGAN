import os
import torch
import numpy as np
import cv2
import config
from generator import Generator
from torchvision.transforms import v2 as tv2

if __name__ == '__main__':

    gen = Generator(in_channels=3).to(config.DEVICE)
    checkpoint = torch.load('gen.pth.tar', map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()

    webcam = cv2.VideoCapture(0)
    cv2.namedWindow("cam-input")
    cv2.namedWindow("cam-output")
    
    while True:
        success, input_image = webcam.read()
        if not success:
            print("Could not get an image. Please check your video source")
            break
        
        
        width = input_image.shape[1]
        height = input_image.shape[0]

        if width >= height:
            d = width - height
            input_image = input_image[:, d//2:d//2+height, :]
        else:
            d = height - width
            input_image = input_image[d//2:d//2+width, :, :]

        input_image = cv2.resize(input_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        cv2.imshow("cam-input", input_image)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.asarray([input_image])
        input_image = np.transpose(input_image, (0, 3, 1, 2))

        input_image = torch.FloatTensor(input_image).to(config.DEVICE)
        
        input_image /= 255.0
        input_image = tv2.functional.normalize(input_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        output_image = gen(input_image)

        output_image = output_image.detach()
        output_image = output_image*0.5 + 0.5
        output_image = tv2.functional.to_dtype(output_image, torch.uint8, scale=True)
        output_image = output_image[0, :, :, :]        
        output_image = output_image.cpu().numpy()
        
        output_image = np.transpose(output_image, (1, 2, 0))
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("cam-output", output_image)

        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

    cv2.destroyWindow("cam-input")
    cv2.destroyWindow("cam-output")