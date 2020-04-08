"""
Flask backend for UCI's COVID-19 anti-misinformation project.
"""
import argparse
import json
import logging
import sys

import flask
from gevent.pywsgi import WSGIServer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel = logging.INFO
logger.addHandler(handler)
logger.propagate = False


def main(port: int):
    app = make_app()
    http_server = WSGIServer(('0.0.0.0', port), app, log=logger, error_log=logger)
    logger.info('Server started on port %i.', port)
    http_server.serve_forever()


def make_app():
    app = flask.Flask(__name__)

    @app.route('/predict/', methods=['POST'])
    def predict():
        data = flask.request.get_json()
        logger.info('request: %s', json.dumps(data))

        prediction = {}  # Best model prediction ever!

        return flask.jsonify(prediction)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2020, help='port to serve the backend on')
    args = parser.parse_args()

    main(port=args.port)