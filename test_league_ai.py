"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
from subprocess import call
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

import cv2

from YAD2K.yad2k.models.keras_yolo import yolo_eval, yolo_head

from retrain_yolo import create_model

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model. Choose')

parser.add_argument(
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model',
    default='YAD2K/model_data/yolo.h5')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='YAD2K/model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='YAD2K/model_data/league_classes.txt')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

parser.add_argument(
    '-out',
    '--output_path',
    type=str,
    help='path to output test images')

subparsers = parser.add_subparsers(dest='subcommand')

vod_option = subparsers.add_parser('mp4')
vod_option.add_argument(
    '-mp4',
    '--test_mp4_vod_path',
    type=str,
    help='path to VOD to analyze. Note - only 1080p videos are allowed!'
)

image_option = subparsers.add_parser('images')
image_option.add_argument(
    '-images',
    '--test_images_path',
    type=str,
    help='path to images to test. These images MUST be size 1920x1080')

args = parser.parse_args()

model_path = os.path.expanduser(args.model_path)
assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
anchors_path = os.path.expanduser(args.anchors_path)
classes_path = os.path.expanduser(args.classes_path)
output_path = os.path.expanduser(args.output_path)

if args.subcommand == 'images':
    test_images_path = os.path.expanduser(args.test_images_path)

if args.subcommand == 'mp4':
    test_mp4_vod_path = os.path.expanduser(args.test_mp4_vod_path)

if not os.path.exists(output_path):
    print('Creating output path {}'.format(output_path))
    os.mkdir(output_path)

sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

# yolo_model = load_model(model_path)
yolo_model, _ = create_model(anchors, class_names)
yolo_model.load_weights('trained_stage_3_best.h5')

# Verify model, anchors, and classes are compatible
num_classes = len(class_names)
num_anchors = len(anchors)
# TODO: Assumes dim ordering is channel last
model_output_channels = yolo_model.layers[-1].output_shape[-1]
assert model_output_channels == num_anchors * (num_classes + 5), \
    'Mismatch between model and given anchor and class sizes. ' \
    'Specify matching anchors and classes with --anchors_path and ' \
    '--classes_path flags.'
print('{} model, anchors, and classes loaded.'.format(model_path))

# Check if model is fully convolutional, assuming channel last order.
model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

# Generate output tensor targets for filtered bounding boxes.
# TODO: Wrap these backend operations with Keras layers.
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=args.score_threshold,
    iou_threshold=args.iou_threshold)

# Save the output into a compact JSON file.
outfile = open('output/game_data.json', 'w')
# This will be appended with an object for every frame.
data_to_write = []


def test_yolo(image, image_file_name):
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), image_file_name))

    # Write data to a JSON file located within the 'output/' directory.
    # This ASSUMES that the game comes from a spectated video starting at 0:00
    # Else, data will not be alligned!
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    data = {}
    data['timestamp'] = '0:00'
    data['champs'] = {}

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        # Save important data to JSON.
        data['champs'][predicted_class] = score

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)


        del draw

    image.save(os.path.join(output_path, image_file_name), quality=90)

def process_mp4(test_mp4_vod_path):
    video = cv2.VideoCapture(test_mp4_vod_path)
    print("Opened ", test_mp4_vod_path)
    print("Processing MP4 frame by frame")

    # forward over to the frames you want to start reading from.
    # manually set this, fps * time in seconds you wanna start from
    video.set(1, 0);
    success, frame = video.read()
    count = 0
    file_count = 0
    success = True
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Loading video %d seconds long with FPS %d and total frame count %d " % (total_frame_count/fps, fps, total_frame_count))

    while success:
        success, frame = video.read()
        if not success:
            break
        if count % 1000 == 0:
            print("Currently at frame ", count)

        # i save once every fps, which comes out to 1 frames per second.
        # i think anymore than 2 FPS leads to to much repeat data.
        if count %  fps == 0:
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            test_yolo(im, str(file_count) + '.jpg')
            file_count += 1
        count += 1

def _main():
    if args.subcommand == 'images':
        for image_file_name in os.listdir(test_images_path):
            try:
                image_type = imghdr.what(os.path.join(test_images_path, image_file_name))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue

            image = Image.open(os.path.join(test_images_path, image_file_name))
            test_yolo(image, image_file_name)

    if args.subcommand == 'mp4':
        process_mp4(test_mp4_vod_path)

    sess.close()

if __name__ == '__main__':
    _main()
