import numpy as np
from PIL import Image
import os
import sys
import json   

def load_json_to_dict():
    json_file = json.load(open('./data/trainables/data.json'))
    return json_file


def get_boundingboxes_and_images_for_chunk(chunk):
    image = []
    boxes = []

    for sample in chunk:

        print("Getting sample: " + str(sample['imageNum']))
        boundingboxes = []
        img = Image.open('./data/trainables/' + str(sample['imageNum']) + '.png')
        img = np.array(img, dtype=np.uint8)
        image.append(img)

        for minion in sample['minionData']:
            bounding_box = minion['minionBB']
            boundingboxes.append(np.array(bounding_box))
        boxes.append(np.array(boundingboxes))

    return boxes,image
        
if __name__ == '__main__':
    data_segments = 400
    data = load_json_to_dict()
    data = [data[x:x+data_segments] for x in range(0, len(data), data_segments)]

    num_chunk = 0

    for chunk in data:
        bounds,images = get_boundingboxes_and_images_for_chunk(chunk)
        
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

        np.savez('./data/train/data_training_set_cluster_' + str(num_chunk),force_zip64=True,images=train_images,boxes=train_boxes)
        np.savez('./data/test/data_test_set_cluster_' + str(num_chunk),images=test_images,boxes=test_boxes)
        np.savez('./data/val/data_val_set_cluster_' + str(num_chunk),images=val_images,boxes=val_boxes)

        print("Saved @ # " + str(num_chunk))
        num_chunk += 1