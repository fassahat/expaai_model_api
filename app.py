from flask import Flask, request, Response
import json
import time
from transformers import pipeline

app = Flask(__name__)

@app.route('/hello', methods=['GET', 'POST'])
def welcome():
    return "Hello World!!"

@app.route('/sentiment_predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentences = data['sentences']
    # time.sleep(1) # we will call model somewhere here
    # result = { "advantages": [sentences[11]], "problems": [sentences[13]], "solutions": [sentences[15]] }
    # print(result)
    classifier = pipeline("text-classification", "fassahat/distill-bert-finetuned-150k-patent-sentences", framework="tf")
    result = classifier(sentences)
    final_result = {'advantages':[], 'solutions':[], 'problems':[]}
    for i in range(len(sentences)):
        if (result[i]['score'] >= 0.8 and len(sentences[i]) > 10):
            if (result[i]['label'] == '0'):
                final_result['solutions'].append(sentences[i])
            elif (result[i]['label'] == '1'):
                final_result['advantages'].append(sentences[i])
            else:
                final_result['problems'].append(sentences[i])
    
    return Response(json.dumps(final_result),  mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)