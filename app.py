from flask import Flask,render_template, request
import numpy as np
import pandas as pd
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.logger import logging
application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict_news():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            text = str(request.form.get('news_text'))
        )

        pred = data.get_as_dataframe()
        print(pred)
        logging.info(f'The dataframe is given as {pred}')

        pred_pipeline = PredictPipeline()
        results = pred_pipeline.predict(pred)
        return render_template('home.html',results=results)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)