import numpy as np
import pandas as pd
import scipy.stats as st
from collections import Counter, defaultdict
import matplotlib as mpl
import os
import pickle
import itertools
import random
import functools
import cv2
import PIL.Image as Image
import matplotlib.pylab as plt
from multiprocessing import Pool
import functools

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

with open('ob_name_lookup.pkl', 'rb') as f:
    obj_name_lookup = pickle.load(f)

flat_image_dir = '/Users/schwenk/wrk/pqa/ade20k/ADE20K_2016_07_26/images/training/flattened/'

question_filters = {
    'dynamics': {
        'object_properties_needed': 'movable',
        'templates': {
            '1_moves_how': 'If the force shown is applied to the {}, how will it move?',
            '1_moves_where': 'If the force shown is applied to the {}, where will it end up?',
            '1_apply_to_move': 'If we want the {} to end up in the region shown, how should we apply a force?'
        }
    },
    'metrics': {
        'object_properties_needed': 'metric',
        'templates': {
            '2_dist_between': 'How far is the {} from the {}?',
            '1_farthest_from': 'What is the farthest object from the {}?',
            '1_closest_to': 'What is the closest object to the {}?',
            '3_maximize_seperation': 'Where should the {} be moved such that is equally far from the {} and the {}'
        }
    },
    'volume': {
        'object_properties_needed': 'container',
        'templates': {
            '2_can_fit': 'Can the {} fit inside the {}?',
            '1_largest_fit': 'What is the largest object in the scene that will fit inside the {}?',
            '2_how_many_fit': 'How many of the {} could fit inside the {}?'
        }
    },
    'liquid': {
        'object_properties_needed': 'liquid',
        'templates': {
            '2_can_liquid_fit': 'Can the {} hold the liquid contained in the {}?',
            '1_est_volume': 'How much liquid could the {} hold?',
        }
    }
}


def draw_obj_sample(obj_annotations, sample_size=5):
    scene_image = Image.open(flat_image_dir + obj_annotations['imageName'])
    plt.imshow(scene_image)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    rectangles = []
    color = []
    for obj in np.random.choice(list(obj_annotations['objects'].values()), sample_size):
        box_color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        rect_ll = obj['rectangle'][0], obj['rectangle'][1]
        rect_w = obj['rectangle'][2] - rect_ll[0]
        rect_h = obj['rectangle'][3] - rect_ll[1]
        rectangles.append(Rectangle(rect_ll, rect_w, rect_h))
        color.append(box_color)
        p = PatchCollection(rectangles, facecolor=color, linewidths=0, alpha=0.075)
        ax.add_collection(p)
        p = PatchCollection(rectangles, facecolor="none", edgecolors=color, linewidths=2)
        ax.add_collection(p)
        ax.text(*rect_ll, obj['classDescription'], color='g')


def draw_question(obj_annotations, involved_objects, question_text):
    scene_image = Image.open(flat_image_dir + obj_annotations['imageName'])
    plt.imshow(scene_image)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_autoscale_on(False)
    rectangles = []
    color = []
    for inv_obj in [obj for obj in obj_annotations['objects'].values() if obj['globalID'] in involved_objects]:
        box_color = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        rect_ll = inv_obj['rectangle'][0], inv_obj['rectangle'][1]
        rect_w = inv_obj['rectangle'][2] - rect_ll[0]
        rect_h = inv_obj['rectangle'][3] - rect_ll[1]
        rectangles.append(Rectangle(rect_ll, rect_w, rect_h))
        color.append(box_color)
        p = PatchCollection(rectangles, facecolor=color, linewidths=0, alpha=0.075)
        ax.add_collection(p)
        p = PatchCollection(rectangles, facecolor="none", edgecolors=color, linewidths=2)
        ax.add_collection(p)
    ax.text(10, 30, question_text, color='g')


def generate_questions_for_image(image_annotation, label_assignments):
    objs_by_class = defaultdict(list)
    for anno in image_annotation['objects'].values():
        objs_by_class[anno['shortName']].append(anno['globalID'])
    generated_questions = defaultdict(dict)
    q_count = 0
    for obj_prop, q_defs in question_filters.items():
        types_present_participating = label_assignments[q_defs['object_properties_needed']].intersection(objs_by_class.keys())
        objects_participating = [ob for k, v in objs_by_class.items() if k in types_present_participating for ob in v]
        q_count += 1
        for q_type, q_template in q_defs['templates'].items():
            qid = 'q_' + q_type + '_' + str(q_count)
            obs_needed = int(q_type[0])
            if len(objects_participating) < obs_needed:
                continue
            chosen_objects = np.random.choice(objects_participating, obs_needed, replace=False)
            classes_of_chosen = [image_annotation['objects'][int(gid.split('_')[-1])]['shortName'] for gid in chosen_objects]
            generated_questions[qid]['question_text'] = q_template.format(*classes_of_chosen)
            generated_questions[qid]['gids'] = chosen_objects
            generated_questions[qid]['obj_classes'] = classes_of_chosen
    return generated_questions


def build_object_entry(image_name, image_path, obj_name_lookup):
    img_seg_name = image_name.replace('.jpg', '_seg.png')
    pixel_array = cv2.imread(os.path.join(image_path, img_seg_name))
    objects = extract_objects(pixel_array, image_name, obj_name_lookup)
    return objects


def build_image_entry(image_df_row, dpp, obj_name_lookup):
    image_fields = {}
    image_name = image_df_row['filename']
    image_path = '../ade20k/' + image_df_row['folder']
    image_fields['imageName'] = image_name
    image_fields['imageName'] = image_name
    image_fields['scene'] = image_df_row['scene']
    image_fields['setting'] = image_df_row['setting']
    image_fields['objects'] = build_object_entry(image_name, image_path, obj_name_lookup)
    dpp[image_name] = image_fields


def call_apply_fn(df):
    global obj_name_lookup
    ds_p = {}
    build_image_entry_part = functools.partial(build_image_entry)
    df.apply((lambda x: build_image_entry_part(x, ds_p, obj_name_lookup)), axis=1)
    return ds_p


def extract_objects(label_array, image_name, obj_name_lookup):
    obj_mask = label_array[::, ::, 0]
    g_chan = label_array[::, ::, 1]
    r_chan = label_array[::, ::, 2]
    class_mask = r_chan.astype(int) / 10 * 256 + g_chan.astype(int)
    obj_anno = defaultdict(dict)
    local_ids = np.unique(obj_mask)
    for obj_id in local_ids:
        obj_pixels = obj_mask == obj_id
        y_indices, x_indices = np.where(obj_pixels == 1)
        obj_anno[obj_id]['rectangle'] = [min(x_indices), min(y_indices), max(x_indices), max(y_indices)]
        obj_anno[obj_id]['objectClass'] = int(class_mask[y_indices[0]][x_indices[0]].item())
        if obj_anno[obj_id]['objectClass'] == 0:
            del obj_anno[obj_id]
            continue
        obj_anno[obj_id]['globalID'] = '_'.join([image_name, str(obj_id)])
        if obj_anno[obj_id]['objectClass'] not in obj_name_lookup:
            del obj_anno[obj_id]
            continue
        obj_anno[obj_id]['classDescription'] = obj_name_lookup[obj_anno[obj_id]['objectClass']]
        obj_anno[obj_id]['shortName'] = obj_name_lookup[obj_anno[obj_id]['objectClass']]
    return {k: v for k, v in obj_anno.items() if v}

