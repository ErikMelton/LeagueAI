import numpy as np
from PIL import Image
import os
import sys
import json

def get_boundingboxes_and_images(jsonfile):
    image = []
    boxes = []

    count = 0
    for sample in jsonfile:
        if count == 400:
            return boxes,image

        print("Getting sample: " + str(sample['imageNum']))
        boundingboxes = []
        img = Image.open('./data/trainables/' + str(sample['imageNum']) + '.png')
        img = np.array(img, dtype=np.uint8)
        image.append(img)

        for minion in sample['minionData']:
            bounding_box = minion['minionBB']
            boundingboxes.append(np.array(bounding_box))
        boxes.append(np.array(boundingboxes))
        count += 1

    return boxes,image
        

label_dict = {'chaos_minion_melee_blue': 1,'chaos_minion_melee_purple': 2,'order_minion_melee_red': 3,'order_minion_melee_blue': 4}

json_file = json.load(open('./data/trainables/data.json'))

dict_to_npz = {}

bounds,images = get_boundingboxes_and_images(json_file)

test_set_cut = int(0.025 * len(images))
val_set_cut = int(0.175 * len(images))
train_set_cut = int(0.80 * len(images))

all_images = np.array(images)
all_boxes = np.array(bounds)

train_images,val_images,test_images = np.split(all_images,[train_set_cut,train_set_cut+val_set_cut])
train_boxes,val_boxes,test_boxes = np.split(all_boxes,[train_set_cut,train_set_cut+val_set_cut])

print("Shape of training images... ", train_images.shape)
print("Shape of val images... ", val_images.shape)
print("Shape of test images... ", test_images.shape)

print("Shape of training boxes... ", train_boxes.shape)
print("Shape of val boxes... ", val_boxes.shape)
print("Shape of test boxes... ", test_boxes.shape)

np.savez('./data/data_training_set_cluster',force_zip64=True,images=train_images,boxes=train_boxes)
np.savez('./data/data_test_set_cluster',images=test_images,boxes=test_boxes)
np.savez('./data/data_val_set_cluster',images=val_images,boxes=val_boxes)