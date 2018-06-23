import json
import numpy as np

from format_minion_anim_files import generateMinionDict,generateScreenshotList
from PIL import Image, ImageDraw

import random

minionDict = generateMinionDict()
screenshotList = generateScreenshotList()
trainDict = []

count = 0

label_dict = {
    'minion_melee_ally': 0,
    'minion_melee_enemy': 1,
    'minion_ranged_ally': 2,
    'minion_ranged_enemy': 3,
    'minion_seige_ally': 4,
    'minion_seige_enemy': 5,
    'minion_super_ally': 6,
    'minion_super_enemy': 7    
}

for screenshot in screenshotList:
    print('Generating training data for image: ' + str(count) + ' of ' + str(len(screenshotList)))
    numMinions = random.randint(4,18)
    minionData = []
    # draw = ImageDraw.Draw(screenshot)

    for miniCount in range(0, numMinions):
        toMergeCategory = random.choice(list(minionDict.items()))
        minion = toMergeCategory[1][random.randint(0,len(toMergeCategory[1])-1)]

        randX = random.randint(0,1280-minion.center[0])
        randY = random.randint(0,720-minion.center[1])
        minionLocationX = int(randX + minion.center[0]) 
        minionLocationY = int(randY + minion.center[1])

        xmin = int(minionLocationX - minion.center[0])
        xmax = int(minionLocationX + minion.center[0])
        ymin = int(minionLocationY - minion.center[1])
        ymax = int(minionLocationY + minion.center[1])
        # draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="yellow")
        minionBoundingBox = np.array([label_dict[minion.minionType],xmin,ymin,xmax,ymax]).tolist()

        minionData.append({'minionType': minion.minionType, 'minionLocation': (minionLocationX,minionLocationY), 'minionBB': minionBoundingBox})
        screenshot.paste(minion.image,(randX,randY),minion.image)

    trainDict.append({'imageNum': count, 'minionData': minionData})
    screenshot.save('./data/trainables/' + str(count) + '.png')
    count += 1

with open('./data/trainables/data.json','w') as file:
    json.dump(trainDict, file)