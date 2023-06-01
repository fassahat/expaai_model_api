# EXPAAI Model - API

It is a flask API which get the requests from the [EXPAAI browser extension](https://github.com/fassahat/expaai_browser_extension_cli) and process them and send back the results to the browser extension. The AI models used in this API are all hosted on [Hugging face hub](https://huggingface.co/). Models that are trained on our datasets can be found in the following [repo](https://huggingface.co/fassahat) on Hugging face hub.

## API Reference

#### Patent Sentiment Analysis

```http
  POST /sentiment_predict
```
This endpoint takes patent sentences and pass these to the model which predicts to which sentiment class each sentence belongs and then send this result back.

#### Patent Semantic Search

```http
  POST /semantic_search
```
This endpoint takes a question and the patent text and pass it to semantic model to get back an answer from the provided text and then send this result back.

## Installation and Deployment

We strongly recommend to use `Anaconda` for this.

1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/).
2. Download/pull the API project and open terminal and go to the project directory.
3. Make a new conda env with [TensorFlow](https://www.tensorflow.org/), intallation details can be found [here](https://docs.anaconda.com/free/anaconda/applications/tensorflow/).
4. Install flask using following command in the new env `conda install flask`.
5. Install transformers using following command in the new env `conda install transformers`.
6. Run the following command to run the API `python app.py`.
