import numpy as np
import pandas as pd
import scipy.stats as st
from collections import Counter, defaultdict
import matplotlib as mpl
import os
import itertools
import random
import cv2
import PIL.Image as Image

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def draw_obj_sample(ax, obj_annotations, sample_size=5):
    ax.set_autoscale_on(False)
    ax.set_aspect('equal')
    rectangles = []
    color = []
    for obj in np.random.choice(list(obj_annotations.values()), sample_size):
        box_color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        rect_ll = obj['rectangle'][0], obj['rectangle'][1]
        rect_w = obj['rectangle'][2] - rect_ll[0]
        rect_h = obj['rectangle'][3] - rect_ll[1]
        rectangles.append(Rectangle(rect_ll, rect_w, rect_h))
        color.append(box_color)
        p = PatchCollection(rectangles, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(rectangles, facecolor="none", edgecolors=color, linewidths=2)
        ax.add_collection(p)
        ax.text(*rect_ll, obj['classDescription'], color='g')


def build_object_entry(image_name, image_path, img_obj_df):
    img_seg_name = image_name.replace('.jpg', '_seg.png')
    pixel_array = cv2.imread(os.path.join(image_path, img_seg_name))
    objects = extract_objs_bboxes(pixel_array, image_name)
    return objects


def build_image_entry(image_df_row, dpp):
    image_fields = {}
    image_name = image_df_row['filename']
    image_path = '../ade20k/' + image_df_row['folder']
    image_fields['imageName'] = image_name
    image_fields['imageName'] = image_name
    image_fields['scene'] = image_df_row['scene']
    image_fields['setting'] = image_df_row['setting']
    image_fields['objects'] = build_object_entry(image_name, image_path, img_obj_df)
    dpp[image_name] = image_fields


def call_apply_fn(df):
    ds_p = {}
    build_image_entry_part = functools.partial(build_image_entry)
    df.apply((lambda x: build_image_entry_part(x, ds_p)), axis=1)
    return ds_p


def extract_objects(label_array, image_name, obj_name_lookup):
    obj_mask = label_array[::, ::, 0:1].reshape(label_array.shape[:2])
    g_chan = label_array[::, ::, 1:2]
    r_chan = label_array[::, ::, 2:3]
    class_mask = r_chan.astype(int) / 10 * 256 + g_chan.astype(int)
    # class_mask = class_mask.astype(int)

    obj_anno = defaultdict(dict)
    local_ids = np.unique(obj_mask.flatten())
    for obj_id in local_ids:
        obj_pixels = obj_mask == obj_id
        y_indices, x_indices = np.where(obj_pixels == 1)
        obj_anno[obj_id]['rectangle'] = [min(x_indices), min(y_indices), max(x_indices), max(y_indices)]
        obj_anno[obj_id]['objectClass'] = int(class_mask[y_indices[0]][x_indices[0]].item())
        if obj_anno[obj_id]['objectClass'] == 0:
            del obj_anno[obj_id]
            continue
        obj_anno[obj_id]['globalID'] = '_'.join([image_name, str(obj_id)])
        obj_anno[obj_id]['classDescription'] = obj_name_lookup[obj_anno[obj_id]['objectClass']]
    return obj_anno

