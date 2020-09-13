import unittest
from semantic_sh import SemanticSimHash


class SemanticSimHashBaseTest:

    class BaseTest(unittest.TestCase):

        def setUp(self):
            self.sh = None
            self.docs = ['Super Mario Bros., one of the most influential and best-selling video games of all time, was '
                         'first released for the Nintendo Entertainment System in Japan.',
                         'Scientists confirm the first detection, using gravitational-wave astronomy, of a '
                         'black hole in the upper mass gap.',
                         'As of 13 September 2020, more than 28.7 million cases have been reported worldwide; more '
                         'than 920,000 people have died and more than 19.4 million have recovered.',
                         'Scientists confirm the first detection, using gravitational-wave astronomy, of a '
                         'black hole in the upper mass gap.'
                         ]

        def test_get_encoding(self):
            embs = self.sh._get_encoding(self.docs)

            self.assertEqual(embs.shape, (self.sh.dim, len(self.docs)))

        def test_get_hash(self):
            hashes = self.sh.get_hash(self.docs)
            self.assertEqual(len(hashes), len(self.docs))
            self.assertEqual(hashes[1], hashes[3])

        def test_get_add_document(self):
            hashes = self.sh.add_document(self.docs)
            self.assertEqual(len(hashes), len(self.docs))

        def get_distance(self):
            texts = ['He was going to the cinema.', 'He went to the cinema.', 'EU leaders will meet about this issue.']
            low_dist, high_dist = self.sh.get_distance(texts[0], texts[1]), self.sh.get_distance(texts[0], texts[2])

            self.assertGreater(high_dist, low_dist)


class SemanticSimHashBertTest(SemanticSimHashBaseTest.BaseTest):

    def setUp(self):
        super().setUp()
        self.sh = SemanticSimHash(model_type='bert-base-cased', dim=768)


if __name__ == '__main__':
    unittest.main()
