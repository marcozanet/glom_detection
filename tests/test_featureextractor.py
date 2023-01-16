
import sys
sys.path.append("/Users/marco/yolo/code")
import unittest
from helical.prognosis.feature_extractor import FeatureExtractor


FOLDER = '/Users/marco/datasets/muw_exps/detection/train/labels'

class TestFeatureExtractor(unittest.TestCase):

    def test_get_aggregator(self):

        extractor = FeatureExtractor(folder=FOLDER, confidence=False)
        extractor.get_aggregator(aggregator='mean')

        return




if __name__ == '__main__':
    unittest.main()
