from flask import Flask, render_template, request

import pickle


app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def hello_world():  # put application's code here

    return render_template('main.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    userInput = request.form.get('userInput')
    formatted_input = vec.transform([userInput])
    output = model.predict(formatted_input).flat[0]
    return render_template('main.html', pred= output)

if __name__ == '__main__':
    app.run()
