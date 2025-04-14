# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import os
import configparser
import time


class Params:
    def __init__(self, path, profile='ORIGINAL'):
        assert os.path.exists(path), 'Cannot find configuration file: {}'.format(path)
        self.path = path

        config = configparser.ConfigParser()

        config.read(self.path)
        params = config[profile]
        self.path = params.get('path')
        self.model = params.get('model', 'fb1')
        self.weights = params.get('weights')
        self.ball_threshold = params.getfloat('ball_threshold', fallback=0.7)
        self.player_threshold = params.getfloat('player_threshold', fallback=0.7)
        self.device = params.get('device', fallback='cuda:0')
        self.camera = params.getint('camera', fallback=6)
        self.annotations_file = params.get('annotations_file')
        if self.annotations_file is None:
            self.annotations_file = 'annotations_' + str(self.camera) + '_' + time.strftime("%Y%m%d_%H%M") + '.pkl'

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.path), "Cannot access path file: {}".format(self.path)
        assert os.path.exists(self.weights), "Cannot access weights file: {}".format(self.weights)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))
        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")
