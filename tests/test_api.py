import unittest
from server import init_app


class APITest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(APITest, self).__init__(*args, **kwargs)

        app_kwargs = {'model_type': 'bert-base-cased', 'dim': 768}
        self.client = init_app(**app_kwargs).test_client()

        self.data = {
            'documents': ['Super Mario Bros., one of the most influential and best-selling video games of all time,'
                          ' was first released for the Nintendo Entertainment System in Japan.',
                          'Scientists confirm the first detection, using gravitational-wave astronomy, of a '
                          'black hole in the upper mass gap.',
                          'As of 13 September 2020, more than 28.7 million cases have been reported worldwide; more '
                          'than 920,000 people have died and more than 19.4 million have recovered.',
                          ]
        }

    def test_generate_hash(self):
        resp = self.client.post('/api/hash', json=self.data)
        hashes = resp.get_json().get('hashes')

        num_docs = len(self.data['documents'])
        self.assertEqual(len(hashes), num_docs)

    def test_add(self):
        resp = self.client.post('/api/add', json=self.data)
        docs = resp.get_json().get('documents')
        hashes, doc_ids = zip(*[(doc['hash'], doc['id']) for doc in docs])

        num_docs = len(self.data['documents'])
        self.assertEqual([i for i in range(num_docs)], list(doc_ids))
        self.assertEqual(len(hashes), num_docs)

    def test_get_distance(self):
        data = {'src': self.data['documents'][0],
                'tgt': self.data['documents'][1]}
        resp = self.client.post('/api/distance', json=data)

        dist = resp.get_json().get('distance')

        self.assertIsInstance(dist, int)

    def test_get_text(self):
        self.client.post('/api/add', json=self.data)
        resp = self.client.get('/api/text/1')
        self.assertEqual(resp.get_data().decode('utf-8'), self.data['documents'][1])
