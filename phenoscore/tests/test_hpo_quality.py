import unittest
from phenoscore.phenoscorer import PhenoScorer


class HPOQualityTester(unittest.TestCase):

    def setUp(self):
        ps = PhenoScorer("TEST", "hpo")
        self.sim = ps._simscorer

    def test_good_neurodevelopmental_hpo_set(self):
        hpo_terms = [
            'HP:0001263', 'HP:0002020', 'HP:0011968',
            'HP:0002099', 'HP:0001631', 'HP:0001270', 'HP:0000750',
            'HP:0002607', 'HP:0000486', 'HP:0031014', 'HP:0033454'
        ]

        result = self.sim.check_hpo_quality(hpo_terms)

        self.assertIsInstance(result, (int, float))
        self.assertGreater(result, 0)
        print("Correct HPO set passed")

    def test_insufficient_hpo_quality(self):
        hpo_terms = [
            'HP:0001263', 'HP:0012758', 'HP:0001270', 'HP:0000750',
            'HP:0000718', 'HP:0001156', 'HP:0001182', 'HP:0030084'
        ] 

        with self.assertRaises(ValueError):
            self.sim.check_hpo_quality(hpo_terms)
        print("ValueError raised correctly")

    def test_no_neurodevelopmental_overlap(self):
        hpo_terms = [
            'HP:0000325', 'HP:0000384', 'HP:0001508', 
            'HP:0002575', 'HP:0000601', 'HP:0001537',
            'HP:0040092', 'HP:0000218', 'HP:0001631',
            'HP:0004691', 'HP:0009611', 'HP:0001643'
            
        ] 

        with self.assertRaises(ValueError):
            self.sim.check_hpo_quality(hpo_terms)
        print("ValueError raised correctly")

if __name__ == "__main__":
    unittest.main()
