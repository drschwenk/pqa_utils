import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os
import pickle
import random
import functools
import cv2
import PIL.Image as Image
import matplotlib.pylab as plt
from multiprocessing import Pool
import functools
import jinja2

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

with open('ob_name_lookup.pkl', 'rb') as f:
    obj_name_lookup = pickle.load(f)

object_properties = pd.read_csv('./object_annotations.csv')
label_assignments = defaultdict(set)
for label_type in object_properties.iloc[:, 1:].columns.tolist():
    label_assignments[label_type].update(
        object_properties[object_properties[label_type] == 1]['shortname'].tolist())

inv_label_assigments = defaultdict(list)
for k, vals in label_assignments.items():
    for v in vals:
        inv_label_assigments[v].append(k)

flat_image_dir = '/Users/schwenk/wrk/pqa/ade20k/ADE20K_2016_07_26/images/training/flattened/'
html_out_path = './test_ds_indoor_sample/gen_html/'
image_out_path = './test_ds_indoor_sample/images/'

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
            # '3_maximize_seperation': 'Where should the {} be moved such that is equally far from the {} and the {}'
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


page_html_template = """
<!DOCTYPE html>
<html>
  <head>
    <style type="text/css">
       .container {
          }
    </style>
  </head>
  <body style=max-width: 100px>
    <div class="container">
      <h1>Question Type: {{question_type}}</h1>
      <ul>
        {% for entry in question_data %}
            <h3>Q{{entry.0}}</h3>
            <p>{{entry.1}}</p>
            <p>{{entry.2}}</p>
            <p>{{entry.3}}</p>
            <p>{{entry.4}}</p>
            <p>{{entry.5}}</p>
            <p></p>
        {% endfor %}
      </ul>
    </div>
    <script src="http://code.jquery.com/jquery-1.10.2.min.js"></script>
    <script src="http://netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"></script>
  </body>
</html>
"""


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
    red_box = (1, 0, 0)
    green_box = (0, 1, 0)
    blue_box = (0, 0, 1)
    color_choices = [red_box, green_box, blue_box]
    for inv_obj in [obj_annotations['objects'][int(obj.split('_')[-1])] for obj in involved_objects]:
        rect_ll = inv_obj['rectangle'][0], inv_obj['rectangle'][1]
        rect_w = inv_obj['rectangle'][2] - rect_ll[0]
        rect_h = inv_obj['rectangle'][3] - rect_ll[1]
        rectangles.append(Rectangle(rect_ll, rect_w, rect_h))
    p = PatchCollection(rectangles, facecolor="none", edgecolors=color_choices, linewidths=2)
    ax.add_collection(p)


def generate_questions_for_image(image_annotation, label_assignments):
    objs_by_class = defaultdict(list)
    for obj_anno in image_annotation['objects'].values():
        if not object_size_filter(obj_anno):
            continue
        objs_by_class[obj_anno['shortName']].append(obj_anno['globalID'])
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
            generated_questions[qid]['gids'] = chosen_objects.tolist()
            generated_questions[qid]['obj_classes'] = classes_of_chosen
            generated_questions[qid]['q_type'] = q_type
            # generated_questions[qid]['objects']
    return generated_questions


def build_object_entry(image_name, image_path, obj_name_lookup):
    img_seg_name = image_name.replace('.jpg', '_seg.png')
    pixel_array = cv2.imread(os.path.join(image_path, img_seg_name))
    objects = extract_objects(pixel_array, image_name, obj_name_lookup)
    img_area = pixel_array.shape[:2]
    return objects, img_area


def build_image_entry(image_df_row, dpp, obj_name_lookup):
    image_fields = {}
    image_name = image_df_row['filename']
    image_path = '../ade20k/' + image_df_row['folder']
    image_fields['imageName'] = image_name
    image_fields['imageName'] = image_name
    image_fields['scene'] = image_df_row['scene']
    image_fields['setting'] = image_df_row['setting']
    image_fields['objects'], image_fields['size'] = build_object_entry(image_name, image_path, obj_name_lookup)
    dpp[image_name] = image_fields


def call_apply_fn(df):
    global obj_name_lookup
    ds_p = {}
    build_image_entry_part = functools.partial(build_image_entry)
    df.apply((lambda x: build_image_entry_part(x, ds_p, obj_name_lookup)), axis=1)
    return ds_p


def complexity_filter(image, min_count=10):
    return len(image['objects']) >= min_count


def setting_filter(image, setting='indoor'):
    return image['setting'] == setting


def scene_filter(image, desired_scenes):
    return image['scene'] in desired_scenes


def object_size_filter(obj_anno, size_thresh=1000):
    rect = obj_anno['rectangle']
    obj_area = (rect[3] - rect[1]) * (rect[2] - rect[0])
    return obj_area > size_thresh


def filter_image(image, filters=set([complexity_filter, setting_filter])):
    for selection_filter in filters:
        if not selection_filter(image):
            return False
    return True


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


def make_question_data(being_asked, image_path, obj_classes, q_num):
    nested_text = []
    image_rel_path = '../' + '/'.join(image_path.split('/')[2:])
    image_link = '<img src="' + image_rel_path + '" width=1000px>'
    question_types = defaultdict(list)
    for q_type, props in question_filters.items():
        for obj in obj_classes:
            if props['object_properties_needed'] in inv_label_assigments[obj]:
               question_types[obj].extend(props['templates'].values())
    nested_text.extend([q_num, image_link, being_asked, obj_classes] + list(question_types.items()))
    return nested_text


def make_page_html(q_data, page_html, q_type):
    j2env = jinja2.Environment()
    return j2env.from_string(page_html).render(question_data=q_data, question_type=q_type)


def render_random_question(filtered_sample, desired_type):
    img_index = random.randint(0, len(filtered_sample) - 1)
    example_image_anno = list(filtered_sample.values())[img_index]
    image_questions = generate_questions_for_image(example_image_anno, label_assignments)
    filtered_questions = {k: v for k, v in image_questions.items() if v['q_type'] == desired_type}
    if not filtered_questions:
        return None, None, None
    rand_q_indx = random.randint(0, len(filtered_questions) - 1)
    random_question = list(filtered_questions.values())[rand_q_indx]
    draw_question(example_image_anno, random_question['gids'], random_question['question_text'])
    image_path = image_out_path + example_image_anno['imageName'].replace('.jpg', '.png')
    plt.savefig(image_path, transparent=True)
    plt.clf()
    return random_question['question_text'], random_question['obj_classes'], image_path


def make_question_type_sample(filtered_sample, question_type, num_qs_to_make):
    question_data = []
    color_choices = ['red', 'green', 'blue'][::-1]
    reset_color = ' <font color="black">'
    for i in range(num_qs_to_make):
        html_color_tags = [' <font color="{}">'.format(cc) for cc in color_choices]
        question_text, obj_classes, image_path = render_random_question(filtered_sample, question_type)
        if question_text:
            for oc in obj_classes:
                question_text = question_text.replace(' ' + oc, html_color_tags.pop() + oc + reset_color, 1)
            question_data.append(make_question_data(question_text, image_path, obj_classes, i + 1))
        else:
            continue
    q_type_html = make_page_html(question_data, page_html_template, question_type)
    html_out_file = os.path.join(html_out_path, question_type + '.html')
    with open(html_out_file, 'w') as f:
        f.write(q_type_html.encode('ascii', 'ignore').decode('utf-8'))