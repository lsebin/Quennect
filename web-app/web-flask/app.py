import json
from flask import Flask, request

app = Flask(__name__)

@app.route('/api/model')
def get_current_time():
    return {'time': time.time()}

if __name__ == "__main__":
app.run(port=5000)