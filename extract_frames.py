import cv2
import os

from PIL import Image

vidcap = cv2.VideoCapture('data/videos/video.mp4')
success,image = vidcap.read()
count=0

while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,count*1000)
    success,image = vidcap.read()
    cv2.imwrite('data/screenshots/%d.png' % count, image)
    count += 1

# count = 0
# for filename in os.listdir('./data/screenshots'):
#     if filename.endswith('.png'):
#         c = Image.open(os.path.join('./data/screenshots/',filename))
#         d = c.resize((1920,1080))
#         count += 1
#         d.save(os.path.join('./data/screenshots/',filename))