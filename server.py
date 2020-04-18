"""Flask Server"""
import argparse
import json

from flask import Flask, jsonify, request, Response
from classification_server import ClassificationServer, ServerModelError

STATUS_OK = "ok"
STATUS_ERROR = "error"

def start(config_file,
          url_root="./classifier",
          host="0.0.0.0",
          port=5000,
          debug=True):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    app = Flask(__name__)
    app.route = prefix_route(app.route, url_root)
    classification_server = ClassificationServer()
    classification_server.start(config_file)

    @app.route('/hello', methods=['GET'])
    def hello():
        print('hello world')
        return 'Hello World!'

    @app.route('/hello_hab', methods=['POST'])
    def hello_hab():
        if request.headers['Content-Type'] == 'text/plain':
            output = {'msg': 'posted'}
            response = Response(
                mimetype="application/json",
                response=json.dumps(output),
                status=201
            )
            return response

    @app.route('/models', methods=['GET'])
    def get_models():
        out = classification_server.list_models()
        return jsonify(out)

    @app.route('/classify', methods=['POST'])
    def classify():
        file = request.files['file']
        id = request.values.get('id')
        # convert that to bytes
        input = [{'id': int(id), 'input': file.read(), 'src':file.filename}]
        out = {}
        try:
            # classification, scores, n_best, times = classification_server.run(input)
            classification, scores, times = classification_server.run(input)
            assert len(classification) == len(input)
            assert len(scores) == len(input)

            out = [[{"src": input[i]['src'], "tgt": classification[i],
                     # "n_best": n_best,
                     # "urls": downloadimages(translation[i]),
                     "pred_score": scores[i]}
                    for i in range(len(classification))]]

        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        return jsonify(out)

    app.run(debug=debug, host=host, port=port, use_reloader=False,
            threaded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OpenNMT-py REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--url_root", type=str, default="/classifier")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--config", "-c", type=str,
                        default="./available_models/conf.json")
    args = parser.parse_args()
    start(args.config, url_root=args.url_root, host=args.ip, port=args.port,
          debug=args.debug)
