import sys
from core.detector import FaceMaskDetector

sys.path.append('core/detector')

if __name__ == '__main__':
    face_mask_detector = FaceMaskDetector()
    face_mask_detector.video_stream(0, 'Face Mask Detector')
