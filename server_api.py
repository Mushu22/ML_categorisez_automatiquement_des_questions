from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = 1
    print(data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=4125)
