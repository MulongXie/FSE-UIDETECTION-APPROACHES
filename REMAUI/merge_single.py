import json
from glob import glob
from os.path import join as pjoin
from tqdm import tqdm
import cv2
import numpy as np
import multiprocessing


def draw_bounding_box(org, corners, color=(0, 255, 0), line=3, name='img', show=False):
    board = org.copy()
    for i in range(len(corners)):
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color, line)
    if show:
        cv2.imshow(name, cv2.resize(board, (500, 800)))
        cv2.waitKey(0)
    return board


def draw_bounding_box_class(org, corners, compo_class, line=3, show=False, name='img', output=None, non_text=False):
    color_map={'Text':(255,6,6), 'Compo':(6,255,6)}
    board = org.copy()
    for i in range(len(corners)):
        if non_text and compo_class[i] == 'Text':
            continue
        board = cv2.rectangle(board, (corners[i][0], corners[i][1]), (corners[i][2], corners[i][3]), color_map[compo_class[i]], line)
    if show:
        cv2.imshow(name, cv2.resize(board, (500, 900)))
        cv2.waitKey(0)

    if output is not None:
        cv2.imwrite(output, board)
    return board


def save_corners_json(file_path, corners, category, new=True):
    '''
    :param corners: [[col_min, row_min, col_max, row_max]]
    '''
    if not new:
        f_in = open(file_path, 'r')
        components = json.load(f_in)
    else:
        components = {'compos': []}
    f_out = open(file_path, 'w')

    for i in range(len(corners)):
        corner = corners[i]
        c = {'category': category[i], 'column_min': corner[0], 'row_min': corner[1], 'column_max': corner[2], 'row_max': corner[3]}
        components['compos'].append(c)
    json.dump(components, f_out, indent=4)


def is_bottom_or_top(corner, img_shape):
    height, width = img_shape[:2]
    column_min, row_min, column_max, row_max = corner
    if row_max/height < 0.05 or row_min/height > 0.9:
        return True
    return False


def resize_label(bboxes, org_height, resize_height, bias=0):
    bboxes_new = []
    scale = org_height/resize_height
    for bbox in bboxes:
        bbox = [int(b * scale + bias) for b in bbox]
        bboxes_new.append(bbox)
    return bboxes_new


def load_detect_result_json(reslut_file):
    print('Loading %d detection results' % len(reslut_file))

    compos = json.load(open(reslut_file, 'r'))['compos']
    bboxes= []
    for compo in compos:
        bboxes.append([compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']])
    return bboxes


def refine_cv(corners, img_shape):
    new_corners = []
    for corner in corners:
        if not is_bottom_or_top(corner, img_shape):
            new_corners.append(corner)
    return new_corners


def refine_txt(corners, img_shape):
    new_corners = []
    height, width = img_shape[:2]
    for corner in corners:
        column_min, row_min, column_max, row_max = corner
        if (row_max - row_min) / height > 0.15:
            continue
        new_corners.append(corner)
    return new_corners


def incorporate(img, bbox_compos, bbox_text, show=False):
    def merge_two_corners(corner_a, corner_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = corner_a
        (col_min_b, row_min_b, col_max_b, row_max_b) = corner_b

        col_min = min(col_min_a, col_min_b)
        col_max = max(col_max_a, col_max_b)
        row_min = min(row_min_a, row_min_b)
        row_max = max(row_max_a, row_max_b)
        return [col_min, row_min, col_max, row_max]

    corners_compo_refine = []
    compos_class_refine = []

    mark_text = np.full(len(bbox_text), False)
    for a in bbox_compos:
        broad = draw_bounding_box(img, [a])
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        new_corner = None
        text_area = 0
        remain = True
        for i in range(len(bbox_text)):
            b = bbox_text[i]
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            # get the intersected area
            col_min_s = max(a[0], b[0])
            row_min_s = max(a[1], b[1])
            col_max_s = min(a[2], b[2])
            row_max_s = min(a[3], b[3])
            w = np.maximum(0, col_max_s - col_min_s)
            h = np.maximum(0, row_max_s - row_min_s)
            inter = w * h
            if inter == 0:
                continue
            # calculate IoU
            ioa = inter / area_a
            iob = inter / area_b
            iou = inter / (area_a + area_b - inter)
            # print('ioa:%.3f, iob:%.3f, iou:%.3f' %(ioa, iob, iou))
            # draw_bounding_box(broad, [b], color=(255,0,0), line=2, show=True)
            # text area
            if iou > 0.6:
                new_corner = b
                # new_corner = merge_two_corners(a, b)
                mark_text[i] = True
                break
            if ioa > 0.55:
                remain = False
                break
            # if iob == 1:
            #     mark_text[i] = True
                # text_area += inter

        if not remain:
            continue
        if new_corner is not None:
            corners_compo_refine.append(new_corner)
            compos_class_refine.append('Text')
        elif text_area / area_a > 0.5:
            corners_compo_refine.append(a)
            compos_class_refine.append('Text')
        else:
            corners_compo_refine.append(a)
            compos_class_refine.append('Compo')

    for i in range(len(bbox_text)):
        if not mark_text[i]:
            corners_compo_refine.append(bbox_text[i])
            compos_class_refine.append('Text')

    if show:
        # draw_bounding_box_class(img, corners_compo_refine, compos_class_refine, show=show, name='merged')
        draw_bounding_box(img, corners_compo_refine, show=show, name='merged')

    return corners_compo_refine, compos_class_refine


def merge(input_name, show=False):
    bbox_compos = load_detect_result_json('E:\\Mulong\\Result\\rico\\rico_remaui\\rico_remaui_cv\\' + input_name + '.json')
    bbox_texts = load_detect_result_json('E:\\Mulong\\Result\\ocr\\' + input_name + '.json')
    org = cv2.imread('E:\\Mulong\\Datasets\\gui\\rico\\combined\\all\\' + input_name + '.jpg')

    bbox_texts = refine_txt(bbox_texts, org.shape)
    bbox_compos = resize_label(bbox_compos, org.shape[0], 600)
    bbox_compos = refine_cv(bbox_compos, org.shape)

    board = draw_bounding_box(org, bbox_texts, color=(255, 0, 0), show=show, name='ocr')
    draw_bounding_box(board, bbox_compos, color=(0, 255, 0), show=show, name='all')

    corners_compo_refine, compos_class_refine = incorporate(org, bbox_compos, bbox_texts, show=show)
    draw_bounding_box_class(org, corners_compo_refine, compos_class_refine, name='Nontext', non_text=True, show=True)

    # save_corners_json(pjoin(output_root, '66529' + '_allele.json'), corners_compo_refine, compos_class_refine)


merge('30800', show=True)
