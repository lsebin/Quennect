import json
import time
import pandas as pd
from flask import Flask, request, jsonify
from run_model import run_shap
from run_lime import run_lime, scale_user_input
# from flaskext.mysql import MySQL
# from flask_sqlalchemy import SQLAlchemy
# from sqlalchemy.orm import DeclarativeBase
 
app = Flask(__name__)


@app.route('/api/model')
def get_current_time():
    # cur = mysql.get_db().cursor()
    # cur.execute('SELECT * FROM states')
    # data = cur.fetchall()
    # cur.close()
    
    # print(data)
    
    data=None
    
    from_user = [1 for _ in range(30)]
    
    X_train = pd.read_csv('X_train.csv')
    X_train = X_train.drop('Unnamed: 0', axis=1)
    scaled = scale_user_input(from_user)
    X_test = pd.Series((x for x in scaled[0]))
    
    top_5_features_rec = run_shap(X_test, X_train)
    # print(top_5_features_rec)
    
    predict, features_weight = run_lime(X_test, X_train)
    rec = None
    
    print(predict)
    print(features_weight)
    
    if predict < 0.8:
        rec = top_5_features_rec
    
    return jsonify({'recommedation': rec, 'features': features_weight, 'sql': data})

if __name__ == "__main__":
    app.run(port=5000)