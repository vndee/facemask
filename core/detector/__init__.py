# -*- coding:utf-8 -*-
import cv2
import time
import torch
import argparse
import numpy as np
from common.logger import get_logger
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from core.detector.loader import load_pytorch_model, pytorch_inference

logger = get_logger(__name__)


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
        '''
        Inference method for image
        :param image:
        :return:
        '''
        answers = []
        height, width, channel = image.shape
        _image = cv2.resize(image, self.target_shape) / 255.0
        _image = np.expand_dims(_image, axis=0).transpose((0, 3, 1, 2))

        y_bboxes_output, y_cls_output = self.model.forward(torch.tensor(_image).float())
        y_bboxes_output, y_cls_output = y_bboxes_output.detach().numpy(), y_cls_output.detach().numpy()

        # remove the batch dimension, for batch is always 1 for inference
        y_bboxes = decode_bbox(self.anchor_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]

        # to speed up, do single class NMS, not multiple classes NMS
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms
        keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh)

        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_scores[idx]
            bbox = y_bboxes[idx]

            # clip the coordinate, avoid the value exceed the image boundary
            xmin, ymin = max(0, int(bbox[0] * width)), max(0, int(bbox[1] * height))
            xmax, ymax = min(int(bbox[2] * width), width), min(int(bbox[3] * height), height)

            if self.draw_result is True:
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, f'{self.id2class[class_id]}: {conf}', (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

            answers.append([class_id, conf, xmin, ymin, xmax])

        return answers, image

    def video_stream(self, video_url, video_name):
        video_capture = cv2.VideoCapture(video_url)
        height, width = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        assert video_capture.isOpened(), 'Video open failed'
        status, idx = True, 0

        while status:
            start_stamp = time.time()
            status, frame_raw = video_capture.read()
            frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            read_frame_stamp = time.time()

            if status is True:
                bboxes, frame = self.infer(frame_raw)
                cv2.imshow(video_name, frame[:, :, :: -1])
                inference_stamp = time.time()
                write_frame_stamp = time.time()
                idx = idx + 1

                logger.info(f'[{idx}/{total_frames}] - read frame: {read_frame_stamp - start_stamp}, infer time: {inference_stamp - read_frame_stamp}, write time: {write_frame_stamp - inference_stamp}')

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face_mask_detector = FaceMaskDetector()
    face_mask_detector.video_stream(0, 'Face Mask Detector')
