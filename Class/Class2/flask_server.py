from flask import Flask, request
app = Flask(__name__)


@app.route('/', methods = ['POST'])
def print_logs():
    content = request.get_json()
    print(content)
    return "Hello"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)
