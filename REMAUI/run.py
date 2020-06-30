# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 00:16:19 2017

@author: soumi
"""
# import numpy as np
import time
import cv2
from viewProcessor.Canny import Canny
from viewProcessor.ContourAnalysis import ContourAnalysis
from viewProcessor.ContourAnalysis import ContourInfo
from Utils.Project import Project
from HierarchyInfo import ViewHierarchyProcessor
from Utils.DipCalculator import DipCalculator
from Utils.Resolution import Resolution
from Utils.Profile import Profile
from Utils import Environment
from ocr.TesseractOCR import TesseractOCR
from ocr.TextProcessor import TextProcessor
from Utils.ColorUtil import CColor
from ocr.TextInfo import TextInfo
from Utils import Util
from layout.RootAlignmentLayoutFilter import RootAlignmentLayoutFilter
from layout.RelativeLayoutFilter import RelativeLayoutFilter
from layout.DefaultLayoutCreator import DefaultLayoutCreator
from layout.LayoutCreatorForList import LayoutCreatorForList
from layout.LayoutCreator import LayoutCreator
import layout.LayoutHelper as LayoutHelper
from layout.LayoutFilter import LayoutFilter
from projectUtil import ProjectGenerator
import copy
from Utils import XmlUtil
from Utils import Constants
from resource.DrawableWriter import DrawableWriter
import os
from projectUtil.ProjectInfo import ProjectInfo
import glob
from os.path import join as pjoin
import json
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut

PROJECT_FOLDER = "templates\\uploads\\"


def resize_by_height(org, resize_height):
    w_h_ratio = org.shape[1] / org.shape[0]
    resize_w = resize_height * w_h_ratio
    re = cv2.resize(org, (int(resize_w), int(resize_height)))
    return re


@ func_set_timeout(60)
def generateProject(imageLocation, show=False):
    fileExitst = os.path.isfile(imageLocation)
    if (not fileExitst):
        print("Can't access the file")
        return
    img_color = cv2.imread(imageLocation)
    img_color = resize_by_height(img_color, 600)

    img_gray = copy.deepcopy(img_color)
    if (len(img_color.shape) == 3):
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    profile = Profile(Resolution.XXHDPI, img_gray.shape[1], img_gray.shape[0])

    dipCalculator = DipCalculator(img_color, profile)

    # create a valid project name and package name
    mProjectName = imageLocation.split('\\')[-1][:-4]
    mOutProjectFolder = PROJECT_FOLDER + mProjectName

    # create project info
    filename, file_extension = os.path.splitext(imageLocation)
    mDrawableWriter = DrawableWriter(file_extension, mOutProjectFolder)

    # dilate and find edges in the provided screenshot
    dst_denoised = cv2.fastNlMeansDenoising(img_gray)
    canny = Canny()
    dst_edge = canny.findEdge(img_gray)
    #    project = Project("sample")
    dst_edge_dilate = canny.addDilate(dst_edge)
    contourAnalysis = ContourAnalysis()
    contours = contourAnalysis.findContoursWithCanny(dst_edge_dilate)
    contoursOutput = contourAnalysis.analyze(dst_edge_dilate, contours)

    # do the hierarchy processing
    hierarchyProcessor = ViewHierarchyProcessor(contoursOutput.rootView, img_color, canny)
    hierarchyInfo = hierarchyProcessor.process()

    # use tesseract to detect the text
    tesseractOCR = TesseractOCR(dst_denoised, dipCalculator, "English")
    textProcessor = TextProcessor(img_color, dst_denoised, hierarchyInfo.biMapViewRect, tesseractOCR, dipCalculator)
    # process text to remove invalid texts
    textInfo = textProcessor.processText(CColor.Red)
    # Add text boxes to hierarchy
    hierarchyProcessor.addTextToHierarchy(textInfo)

    # List support right now not implemented
    creator = DefaultLayoutCreator(contoursOutput.rootView, mProjectName, tesseractOCR, mDrawableWriter, img_color,
                                   mOutProjectFolder, dipCalculator)

    # create layout
    creator.createDocument()

    # ****** write out detected compos ******
    annotation = mProjectName + ' '
    compos = creator.mDrawableWriter.drawableInfos
    for key in compos:
        compo = compos[key].rectView
        annotation += '{},{},{},{},{} '.format(str(compo.x), str(compo.y), str(compo.x + compo.width),
                                               str(compo.y + compo.height), str(compo.mType))
    # print(annotation)
    open(pjoin(Output_root, mProjectName + '.txt'), 'w').write(annotation)
    # cv2.imwrite(pjoin(Output_root, mProjectName + '.png'), creator.mImage)

    if show:
        cv2.imshow('det', creator.mImage)
        cv2.waitKey()


if __name__ == "__main__":

    Input_root = 'E:\\Mulong\\Datasets\\gui\\rico\\combined\\all'
    Output_root = 'E:\\Temp\\fse\\remaui-2'
    # Output_root = 'E:\\Mulong\\Result\\rico\\rico_remaui'

    data = json.load(open('E:\\Mulong\\Datasets\\gui\\rico\\instances_test.json', 'r'))
    input_paths_img = [pjoin(Input_root, img['file_name'].split('/')[-1]) for img in data['images']]
    input_paths_img = sorted(input_paths_img, key=lambda x: int(x.split('\\')[-1][:-4]))  # sorted by index

    wrong = 0
    start_index = 0
    end_index = 50000
    index = start_index
    while index < end_index:
        input_path_img = pjoin(Input_root, str(index) + '.jpg')
        if input_path_img not in input_paths_img:
            index += 1
            continue

        start = time.clock()
        print('Processing ', input_path_img)
        try:
            generateProject(input_path_img, show=False)
        except FunctionTimedOut:
            print('*** Time Out ***')
        except:
            print('****** Process Failed: %d ******' % wrong)
            wrong += 1
        print('Time: %.3fs' %(time.clock() - start))
        print(time.ctime(), '\n')

        with open('run_time.txt', 'a') as run_time:
            run_time.write(str(format(time.clock() - start, '.3f')) + ' ')
        index += 20
