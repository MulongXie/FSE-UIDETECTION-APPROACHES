import cv2
import numpy as np
from random import randint as rint
import time
import json
from os.path import join as pjoin
import multiprocessing

import xianyu_ocr as ocr
import xianyu_merge as merge
import xianyu_utils as utils


def gradient_laplacian(org):
    lap = cv2.Laplacian(org, cv2.CV_16S, 3)
    lap = cv2.convertScaleAbs(lap)
    return lap


def rm_noise_flood_fill(img, grad_thresh=10, show=False):
    grad_thresh = (grad_thresh, grad_thresh, grad_thresh)
    mk = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.uint8)
    for x in range(0, img.shape[0], 10):
        for y in range(0, img.shape[1], 10):
            if mk[x, y] == 0:
                cv2.floodFill(img, mk, (y, x), (0,0,0), grad_thresh, grad_thresh, cv2.FLOODFILL_FIXED_RANGE)
    if show:
        cv2.imshow('floodfill', img)
        cv2.waitKey()


def slicing(img, leaves, base_upleft, show=False):
    '''
    slices: [[col_min, row_min, col_max, row_max]]
    '''
    row, col = img.shape[:2]
    slices = []
    up, bottom, left, right = -1, -1, -1, -1
    obj = False
    for x in range(row):
        if np.sum(img[x]) != 0:
            if not obj:
                up = x
                obj = True
                continue
        else:
            if obj:
                bottom = x
                obj = False
                box = [0, up, col, bottom]
                if bottom - up > 10 and (bottom - up) * col > 200:
                    slices.append(box)
                continue

    obj = False
    for y in range(col):
        if np.sum(img[:, y]) != 0:
            if not obj:
                left = y
                obj = True
                continue
        else:
            if obj:
                right = y
                obj = False
                box = [left, 0, right, row]
                if right - left > 10 and (right - left) * row > 200:
                    slices.append(box)
                continue

    for box in slices:
        slice_img = img[box[1]:box[3], box[0]:box[2]]
        box = [box[0] + base_upleft[0], box[1] + base_upleft[1], box[2] + base_upleft[0], box[3] + base_upleft[1]]
        children = slicing(slice_img, leaves, (box[0], box[1]), show=show)
        if len(children) == 0:
            leaves.append(box)
            if show:
                cv2.imshow('slices', slice_img)
                cv2.waitKey()
    return slices


def detect_compo(org, output_path=None, show=False):
    start = time.clock()
    grad = gradient_laplacian(org)
    rm_noise_flood_fill(grad, show=False)
    compo_bbox = []
    slicing(grad, compo_bbox, (0, 0), show=False)
    utils.draw_bounding_box(org, compo_bbox, show=show)
    if output_path is not None:
        utils.save_corners_json(output_path + '.json', compo_bbox, np.full(len(compo_bbox), 'Compo'))
    # print('Compo det [%.3fs]' % (time.clock() - start))
    return compo_bbox


def xianyu(input_path_img, output_path, num, show=False, write_img=False):

    start = time.clock()
    org = cv2.imread(input_path_img)
    img = utils.resize_by_height(org, resize_height=800)

    compo = detect_compo(img, show=show, output_path=output_path.replace('rico_xianyu_bg_ocr', 'rico_xianyu_bg_cv'))
    text = ocr.ocr(org, show=show)
    compo_merge, categories = merge.incorporate(img, compo, text, show=show)

    utils.draw_bounding_box_class(img, compo_merge, categories, output=output_path + '.png' if write_img else None)
    utils.save_corners_json(output_path + '.json', compo_merge, categories)
    print('[%.3fs] %d %s' % (time.clock() - start, num, input_path_img))


input_img_root='E:\\Mulong\\Datasets\\gui\\rico\\combined\\all'
output_root='E:\\Temp\\fse\\xianyu-1'
data = json.load(open('E:\\Mulong\\Datasets\\gui\\rico\\instances_test.json', 'r'))
input_paths_img = [pjoin(input_img_root, img['file_name'].split('/')[-1]) for img in data['images']]
input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

if __name__ == '__main__':
    cpu_num = 3
    pool = multiprocessing.Pool(processes=cpu_num)
    num = 0
    start_index = 809
    end_index = 100000
    for input_path_img in input_paths_img:
        index = input_path_img.split('\\')[-1][:-4]
        if int(index) < start_index:
            continue
        if int(index) > end_index:
            break
        output_path = pjoin(output_root, str(index))
        pool.apply_async(xianyu, (input_path_img, output_path, num, False, ))
        num += 1
    pool.close()
    pool.join()