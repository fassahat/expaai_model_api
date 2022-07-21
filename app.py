from flask import Flask, request, Response
import json
import time
app = Flask(__name__)

@app.route('/hello', methods=['GET', 'POST'])
def welcome():
    return "Hello World!!"

@app.route('/sentiment_predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentences = data['sentences']
    time.sleep(10) # we will call model somewhere here
    result = [sentences[0], sentences[1]]
    return Response(json.dumps(result),  mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)