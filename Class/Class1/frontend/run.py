from flask import Flask, render_template, request
import os
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/run", methods = ['POST'])
def run():
    hardware = request.form['hardware']
    dbname = request.form['database']
    train = request.files['train']
    train.save(os.path.join('uploads',train.filename))
    validation = request.files['validation']
    test = request.files['test']
    epoch = request.form['epoch']
    training = request.form['training']

    print(hardware)
    print(dbname)
    print(train.filename)
    print(validation.filename)
    print(test.filename)
    print(epoch)
    print(training)
    return "Successfully"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug = True )
