import os, io
import time
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import numpy as np
import cv2 as cv

# font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", 16)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'apikey.json'

client = vision.ImageAnnotatorClient()

FILE_NAME = 'image.jpg'
DIR_PATH = '.'
FILE_PATH = os.path.join(DIR_PATH, FILE_NAME)

with io.open(FILE_PATH, 'rb') as image_file:
    content = image_file.read()

img = cv.imread(FILE_PATH)

image = vision.types.Image(content=content)
response = client.text_detection(image=image)
texts = response.text_annotations

for text in texts:
    im_text = text.description
    print(im_text)
    vertices = [[int(vertex.x), int(vertex.y)] for vertex in text.bounding_poly.vertices]
    vertices_array = np.asarray(vertices, dtype=np.int32)
    pts = vertices_array.reshape((-1,1,2))
    img = cv.polylines(img,[pts],True,(0,255,255))
    font = cv.FONT_HERSHEY_PLAIN
    cv.putText(img,im_text,tuple(vertices[-1]), font, 1,(255,0,0),1,cv.LINE_AA)
cv.imwrite('image_detect.jpg', img)
cv.imshow('img', img)
