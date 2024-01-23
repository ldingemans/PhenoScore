import unittest
from phenoscore.facial_feature_extraction.extract_facial_features import QMagFaceExtractor
import pandas as pd
import numpy as np
import os
import ast


class FacialFeatureExtractionTester(unittest.TestCase):
    def setUp(self):
        path_to_script = os.path.realpath(__file__).split(os.sep)[:-2]
        path_to_script.insert(1, os.sep)
        path_to_qmagface = os.path.join(*path_to_script, 'facial_feature_extraction')
        self._qmagface = QMagFaceExtractor(path_to_dir=path_to_qmagface)

    def test_positive_output(self):
        path_to_script = os.path.realpath(__file__).split(os.sep)[:-2]
        path_to_script.insert(1, os.sep)
        path_to_img = os.path.join(*path_to_script, 'sample_data', 'control_1.png')
        assert(type(self._qmagface.process_file(path_to_img)) == list)
        assert(len(self._qmagface.process_file(path_to_img)) == 512)

    def test_negative_output(self):
        random_img = (np.random.standard_normal([28, 28, 3]) * 255).astype(np.uint8)
        assert(self._qmagface.process_file(random_img) is None)
