from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/salary_prediction')
def salary_prediction():
    return render_template('salary_prediction.html')

@app.route('/data_analysis')
def data_analysis():
    return render_template('data_analysis.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
