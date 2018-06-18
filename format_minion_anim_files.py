import os

from PIL import Image

class Minion():
    def __init__(self,minionType,image):
        self.minionType = minionType
        self.image = image
        width,height = image.size
        self.center = ((width/2),(height/2))
        self.bboxwidth = 330
        self.bboxheight = 330

def generateMinionDict():
    minionDict = {'chaos_minion_melee_blue': [],'chaos_minion_melee_purple': [],'order_minion_melee_red': [],'order_minion_melee_blue': []}

    dataDir = './data/'

    for subdir,dirs,files in os.walk(dataDir):
        for file in files:
            if subdir != './data/screenshots' and subdir != './data/trainables' and subdir != './data/videos':
                minionType = subdir[7:]
                minionType = minionType.split('\\', 1)[0]
                minionImage = Image.open(os.path.join(subdir,file))
                minion = Minion(minionType,minionImage)
                minionDict[minion.minionType].append(minion)                

    return minionDict

def generateScreenshotList():
    dataDir = './data/screenshots'
    screenshots = []

    for subdir,dirs,files in os.walk(dataDir):
        for file in files:
            screenshot = Image.open(os.path.join(subdir,file))
            screenshots.append(screenshot)

    return screenshots        