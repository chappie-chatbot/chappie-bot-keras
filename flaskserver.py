from flask import Flask, request
from predictor import Predictor
import os

port = os.environ['PORT'] if 'PORT' in os.environ else 80

app = Flask(__name__)
predictor = Predictor().init().configure().load_model()


@app.route('/message', methods=['POST'])
def post_message():
    msg = request.get_json(force=True)['message']
    return predictor.predict_top_response(msg)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
