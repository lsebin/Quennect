import json
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "home"

@app.route('/api/model')
def model():
    return "model"

if __name__ == "__main__":
    app.run(debug=True)