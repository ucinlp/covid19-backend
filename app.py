"""
Flask backend for UCI's COVID-19 anti-misinformation project.
"""
import argparse
import json
import logging
import sys

import flask
import flask_cors

from backend.ml.bertscore import BertScoreDetector
from backend.ml.misconception import MisconceptionDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel = logging.INFO
logger.addHandler(handler)
logger.propagate = False


app = flask.Flask(__name__)
flask_cors.CORS(app)

# TODO: Periodically reload.
with open('misconceptions.jsonl', 'r') as f:
    app.misconceptions = MisconceptionDataset.from_jsonl(f)
app.detector = BertScoreDetector('bert-base-uncased')


@app.route('/predict/', methods=['POST'])
def predict():
    raw = flask.request.get_data()
    data = json.loads(raw) if raw else {}
    logger.info('request: %s', json.dumps(data))
    prediction = app.detector.predict(data['input'], app.misconceptions)

    return flask.jsonify(prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='ip to listen on')
    parser.add_argument('--port', type=int, default=2020, help='port to serve the backend on')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
