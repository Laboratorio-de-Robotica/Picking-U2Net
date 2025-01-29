from u2net_predict import U2netModel
import argparse

#import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt

print('u2net/u2netp prediction')
print('Processes the provided input image, shows the prediction SOD heatmap, and optionally saves it')

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input image file path", default="images/imagen_13r.jpeg")
parser.add_argument("-o", "--output", help="output image file path", default="")
parser.add_argument("-m", "--model", help="model, either u2net (default) or u2netp", default="u2net")
args = parser.parse_args()

model_name = args.model
input_image_path = args.input
output_image_path = args.output

# Input image ndarray
#input_image = cv.imread(input_image_path)
input_image = io.imread(input_image_path)

model = U2netModel(model_name)
output_image = model(input_image)

if(output_image_path):
    io.imsave(output_image_path, output_image)

#cv.imshow('map', output_image)
#cv.waitKey(0)
plt.imshow(output_image)
plt.show()