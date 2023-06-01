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
    classifier = pipeline("text-classification", "fassahat/anferico-bert-for-patents-finetuned-557k-patent-sentences", framework="tf")
    result = classifier(sentences)
    # print(result)
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

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    data = request.get_json()
    patent_text = data['patent_text']
    question = data['question']

    # question_answerer_roberta = pipeline("question-answering", model='deepset/roberta-base-squad2')
    question_answerer_bert_large = pipeline("question-answering", model='bert-large-uncased-whole-word-masking-finetuned-squad')
    # question_answerer_distilbert = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

    # answer_roberta = question_answerer_roberta(question=question, context=patent_text)
    answer_bert_large = question_answerer_bert_large(question=question, context=patent_text)
    # answer_distilbert = question_answerer_distilbert(question=question, context=patent_text)
    # print('answer_roberta: ', answer_roberta)
    # print('answer_bert_large: ', answer_bert_large)
    # print('answer_distilbert: ', answer_distilbert)

    # result = {}
    
    # if ((answer_roberta['score'] >= answer_bert_large['score']) & (answer_roberta['score'] >= answer_distilbert['score'])):
    #     result = { 'answer' : answer_roberta['answer'] }
    # elif ((answer_bert_large['score'] >= answer_roberta['score']) & (answer_bert_large['score'] >= answer_distilbert['score'])):
    #     result = { 'answer' : answer_bert_large['answer'] }
    # else:
    #     result = { 'answer' : answer_distilbert['answer'] }

    result = { 'answer' : answer_bert_large['answer'] }

    return  Response(json.dumps(result),  mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)