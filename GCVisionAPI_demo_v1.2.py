import os, io
import time
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import numpy as np
import cv2 as cv
import glob
# font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", 16)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'apikey.json'

client = vision.ImageAnnotatorClient()


IN_DIR_PATH = '/root/GCloud_Vision/TestIMGs'
OUT_DIR_PATH = '/root/GCloud_Vision/ResIMGs'
EXT = 'jpg'

def put_text(text, number, cursor, image):
    height, width, channels = image.shape
    font_height = 23
    buff = np.ones((font_height,width,3), np.uint8) * 255
    image = np.concatenate((image, buff), axis=0)
    text = str(number) + '. ' + text
    image = cv.putText(image,text,tuple(cursor), font, 1,(0,0,0),1,cv.LINE_AA)
    cursor[1] += font_height
    return image, cursor

test_images = [f for f in glob.glob(IN_DIR_PATH + "/*." + EXT)]
for image_path in test_images:
    image_name = image_path.split('/')[-1]
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    img = cv.imread(image_path)
    height, width, channels = img.shape
    img2 = np.ones((1,width,3), np.uint8) * 255 
    cursor = [10,20]

    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for i in range(1, len(texts)):
        text = texts[i]
        im_text = text.description
#        print(im_text)
        vertices = [[int(vertex.x), int(vertex.y)] for vertex in text.bounding_poly.vertices]
        vertices_array = np.asarray(vertices, dtype=np.int32)
        pts = vertices_array.reshape((-1,1,2))
        img = cv.polylines(img,[pts],True,(0,255,255))
        font = cv.FONT_HERSHEY_PLAIN
        img = cv.putText(img,str(i),tuple(vertices[-1]), font, 1,(255,0,0),1,cv.LINE_AA)
        img2, cursor = put_text(text=im_text, number=i, cursor=cursor, image=img2)

    img = np.concatenate((img, img2), axis=0)
    save_path = os.path.join(OUT_DIR_PATH, image_name)
#    print(save_path)
    cv.imwrite(save_path, img)
