from semantic_sh import SemanticSimHash
from flask import Flask, request, jsonify
import argparse

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # add utf-8 support
sh = None


def init_app(**kwargs):
    """Initialize model and return app object to use for WSGI container"""
    global sh
    sh = SemanticSimHash(**kwargs)
    return app


@app.route('/api/hash', methods=['POST'])
def generate_hash():
    req_json = request.get_json()
    docs = req_json.get('documents')

    # convert negative hashes to 2's complement representation
    return jsonify({'hashes': [hex(h & (2**sh.key_size-1)) for h in sh.get_hash(docs)]})


@app.route('/api/add', methods=['POST'])
def add():
    req_json = request.get_json()
    hashes, ids = sh.add_document(req_json.get('documents'))
    return jsonify({'documents': [{'id': doc_id, 'hash': hex(h & (2**sh.key_size-1))} for doc_id, h in zip(ids, hashes)]})


@app.route('/api/find-similar', methods=['POST'])
def find_similar():
    req_json = request.get_json()
    similar_texts = sh.find_similar(req_json.get('text'))
    return jsonify({'similar_texts': similar_texts})


@app.route('/api/distance', methods=['POST'])
def get_distance():
    req_json = request.get_json()
    txt0, txt1 = req_json.get('src'), req_json.get('tgt')
    return jsonify({'distance': sh.get_distance(txt0, txt1)})


@app.route('/api/similarity-groups', methods=['GET'])
def get_groups():
    return jsonify([group for group in sh.get_similar_groups()])


@app.route('/api/text/<int:id>', methods=['GET'])
def get_text(id):
    return sh.get_doc_by_id(id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    flask_group = parser.add_argument_group('app')
    flask_group.add_argument('--host', required=False, default='0.0.0.0')
    flask_group.add_argument('--port', required=False, default=80)

    model_group = parser.add_argument_group('model')
    model_group.add_argument('--model-type', required=False, default='fasttext', help='Type of model to run: fasttext or any pretrained model name from huggingface/transformers')
    model_group.add_argument('--model-path', required=False, help='Path to vector files of fasttext models')
    model_group.add_argument('--key-size', required=False, type=int, default=256, help='Hash length in bits')
    model_group.add_argument('--dim', required=False, type=int, default=300, help='Dimension of text representations according to chosen model type')
    model_group.add_argument('--stop-words', nargs='*', help='List of stop words to exclude')

    loader_group = parser.add_argument_group('loader')
    loader_group.add_argument('--load-from', required=False, help='Load previously saved state')

    args = parser.parse_args()

    # Save each argument group to dict to make separately accessible
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {action.dest: getattr(args, action.dest, None) for action in group._group_actions}
        arg_groups[group.title] = group_dict

    fname = arg_groups['loader']['load_from']
    sh = SemanticSimHash.load(fname) if fname else SemanticSimHash(**arg_groups['model'])

    app.run(**arg_groups['app'])
