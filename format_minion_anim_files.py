import os
import numpy as np

from PIL import Image

class Minion():
    def __init__(self,minionType,image):
        self.minionType = minionType
        self.image = image
        width,height = image.size
        self.center = ((width/2),(height/2))
        self.bboxwidth = width
        self.bboxheight = height

def generateMinionDict():
    minionDict = {'chaos_minion_melee_blue': [],'chaos_minion_melee_purple': [],'order_minion_melee_red': [],'order_minion_melee_blue': []}

    dataDir = './data/'

    for subdir,dirs,files in os.walk(dataDir):
        for file in files:
            if 'minion' in subdir:
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

def preprocess_minion_files():
    for subdir,dirs,files in os.walk('./data/'):
        for file in files:
            if 'minion' in subdir:
                minionImage = Image.open(os.path.join(subdir,file))
                minionImage.load()
                
                image_data = np.asarray(minionImage)
                image_data_bw = image_data[:,:,3]
                non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
                non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
                cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

                image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
                reduced_minion_image = Image.fromarray(image_data_new)
                reduced_minion_image.save(os.path.join(subdir,file))

if __name__ == '__main__':
    preprocess_minion_files()