import unittest
import topicmodelling.classes
import pandas as pd

testDataRaw = pd.read_csv(r'C:\Users\Tobias Fechner\Documents\1_Uni\fyp\git_repo_fyp\data\apiIntegrations.ga\energy_environment_sustainableenergy_electricvehicles.csv', index_col=0)

class TestCorpus(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.corpus = topicmodelling.classes.Corpus(name='test_get_bow')

    def testGetBOW(self):
        self.corpus.data = testDataRaw
        self.corpus.getBagsOfWords()

        self.assertIsInstance(self.corpus.data, pd.DataFrame)
        self.assertIn('bow', self.corpus.data.columns)
        self.assertTrue(all(self.corpus.data['bow'].apply(lambda x: all(word >= 2 for word in x))))
        self.assertTrue(all(self.corpus.data['bow'].apply(lambda x: all(word >= 2 for word in x))))

    def testUpdateVocab(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
