import json
import time
import pandas as pd
from flask import Flask, request, jsonify
from run_model import run_shap
from run_lime import run_lime, scale_user_input
from apicall import get_analysis

app = Flask(__name__)


@app.route('/api/model')
def get_current_time():
    from_user = [1 for _ in range(30)]
    
    X_train = pd.read_csv('X_train.csv')
    X_train = X_train.drop('Unnamed: 0', axis=1)
    scaled = scale_user_input(from_user)
    X_test = pd.Series((x for x in scaled[0]))
    
    top_5_features_rec = run_shap(X_test, X_train)
    # print(top_5_features_rec)
    
    predict, features_weight, feature_names = run_lime(X_test, X_train)
    rec = None
    
    # print(predict)
    # print(features_weight)
    
    # if predict < 0.8:
    #     rec = top_5_features_rec
    
    rec = top_5_features_rec
    
    raw_dic = {feature_names[i]: from_user[i] for i in range(len(from_user))}
    
    fw_json = '\n'.join([f'"{key}": {value}' for key, value in features_weight.items()])
    rd_json = '\n'.join([f'"{key}": {value}' for key, value in raw_dic.items()])
    r_json = '\n'.join([f'"{key}": {value}' for key, value in rec.items()])
    
    data = {'Explanation': fw_json, 'Raw Data': rd_json, 'Recommendation': r_json}
    
    r_bracket = '{'
    l_bracket = '}'
    latter = '\n'.join([f'{key}:{r_bracket}\n{value}{l_bracket}' for key, value in data.items()])
    formatted_json = f'Prediction : {predict} \n{latter}'
    
    
    txt = get_analysis(formatted_json)
    
    return jsonify({'recommedation': rec, 'features': features_weight, 'response': txt})

if __name__ == "__main__":
    app.run(port=5000)