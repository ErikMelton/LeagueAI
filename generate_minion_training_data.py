import json
import numpy as np

from format_minion_anim_files import generateMinionDict,generateScreenshotList
from PIL import Image, ImageDraw

import random

minionDict = generateMinionDict()
screenshotList = generateScreenshotList()
trainDict = []

count = 0

label_dict = {'chaos_minion_melee_blue': 1,'chaos_minion_melee_purple': 2,'order_minion_melee_red': 3,'order_minion_melee_blue': 4}

for screenshot in screenshotList:
    print('Generating training data for image: ' + str(count) + ' of ' + str(len(screenshotList)))
    numMinions = random.randint(4,18)
    minionData = []

    for miniCount in range(0, numMinions):
        toMergeCategory = random.choice(list(minionDict.items()))
        minion = toMergeCategory[1][random.randint(0,len(toMergeCategory[1])-1)]

        randX = random.randint(0-minion.center[0],1920-minion.center[0])
        randY = random.randint(0-minion.center[1],1080-minion.center[1])
        minionLocationX = randX + minion.center[0] 
        minionLocationY = randY + minion.center[1]

        xmin = (minionLocationX - minion.bboxwidth/2)
        xmax = (minionLocationX + minion.bboxwidth/2)
        ymin = (minionLocationY - minion.bboxheight/2)
        ymax = (minionLocationY - minion.bboxheight/2)
        minionBoundingBox = np.array([label_dict[minion.minionType],xmin,ymin,xmax,ymax]).tolist()

        minionData.append({'minionType': minion.minionType, 'minionLocation': (minionLocationX,minionLocationY), 'minionBB': minionBoundingBox})
        screenshot.paste(minion.image,(randX,randY),minion.image)
    
    trainDict.append({'imageNum': count, 'minionData': minionData})
    screenshot.save('./data/trainables/' + str(count) + '.png')
    count += 1

with open('./data/trainables/data.json','w') as file:
    json.dump(trainDict, file)