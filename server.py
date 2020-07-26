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


@app.route('/api/hash', methods=['GET'])
def generate_hash():
    txt = request.args['text']
    return hex(sh.get_hash(txt))


@app.route('/api/add', methods=['GET'])
def add():
    txt = request.args['text']
    sh.add_document(txt)
    return jsonify(sucess=True)


@app.route('/api/find-similar', methods=['GET'])
def find_similar():
    similar_texts = sh.find_similar(request.args['text'])
    return jsonify(similar_texts)


@app.route('/api/distance', methods=['GET'])
def get_distance():
    txt0, txt1 = request.args['src'], request.args['tgt']
    return str(sh.get_distance(txt0, txt1))


@app.route('/api/similarity-groups', methods=['GET'])
def get_groups():
    return jsonify([group for group in sh.get_similar_groups()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    flask_group = parser.add_argument_group('app')
    flask_group.add_argument('--host', required=False, default='0.0.0.0')
    flask_group.add_argument('--port', required=False, default=80)

    model_group = parser.add_argument_group('model')
    model_group.add_argument('--model-type', required=False, default='fasttext', help=f'Type of model to run: fasttext or any pretrained model name from huggingface/transformers')
    model_group.add_argument('--model-path', required=False, help=f'Path to vector files of fasttext models')
    model_group.add_argument('--key-size', required=False, type=int, default=256, help='Hash length in bits')
    model_group.add_argument('--dim', required=False, type=int, default=300, help='Dimension of text representations according to chosen model type')
    model_group.add_argument('--stop-words', nargs='*', help='List of stop words to exclude')

    args = parser.parse_args()

    # Save each argument group to dict to make separately accessible
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {action.dest: getattr(args, action.dest, None) for action in group._group_actions}
        arg_groups[group.title] = group_dict

    sh = SemanticSimHash(**arg_groups['model'])

    app.run(**arg_groups['app'])
