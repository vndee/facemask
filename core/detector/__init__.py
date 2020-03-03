# -*- coding:utf-8 -*-
import cv2
import time

import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from core.detector.loader import load_pytorch_model, pytorch_inference


class FaceMaskDetector:
    def __init__(self, model_path, conf_thresh, iou_thresh, target_shape, draw_result, show_result):
        '''
        Main class for face mask detection inference
        :param model_path: Path to model weight
        :param conf_thresh: min threshold of classification probability
        :param iou_thresh: IOU threshold of NMS
        :param target_shape: model input size
        :param draw_result: whether to draw bounding box to the image
        :param show_result: whether to display to image
        '''

        self.model = load_pytorch_model(model_path)
        self.feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        self.anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        self.anchor_ratios = [[1, 0.62, 0.42]] * 5
        self.anchors = generate_anchors(self.feature_map_sizes, self.anchor_sizes, self.anchor_ratios)
        self.anchor_exp = np.expand_dims(self.anchors, axis=0)
        self.id2class = {0: 'Mask', 1: 'No Mask'}

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_shape = target_shape
        self.draw_result = draw_result
        self.show_result = show_result

    def infer(self, image):
        answers = []
        height, width, channel = image.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-mode', type=int, default=0, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, show_result=True, target_shape=(260, 260))
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)