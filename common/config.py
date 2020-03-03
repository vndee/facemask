import os
from collections import namedtuple


def get_devenv():
    conf = dict()

    conf['detector_model_path'] = os.environ.get('DETECTOR_MODEL_PATH', 'static/model.vndee')
    conf['detector_conf_thresh'] = os.environ.get('DETECTOR_CONF_THRESH', 0.5)
    conf['detector_iou_thresh'] = os.environ.get('DETECTOR_IOU_THRESH', 0.4)
    conf['detector_target_shape'] = os.environ.get('DETECTOR_TARGET_SHAPE', (260, 260))
    conf['detector_draw_result'] = os.environ.get('DETECTOR_DRAW_RESULT', True)
    conf['detector_show_result'] = os.environ.get('DETECTOR_SHOW_RESULT', True)

    return conf


env = get_devenv()
AppConf = namedtuple('AppConf', env.keys())(*env.values())