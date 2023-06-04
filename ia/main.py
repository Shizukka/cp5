from flask import Flask, request
import tensorflow as tf


UPLOAD_FOLDER = './downloads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
model = None
@app.route('/config', methods=['POST'])
def doConfig():
    global model
    model = tf.keras.models.load_model("best_model.h5", compile=False)
    print("Model loaded !")
    model.summary()
    return {"status": "ok"}


@app.route('/evaluate', methods=['POST'])
def doEvaluate():
    global model

    numbers = request.json['x_data']

    numbers_array = tf.convert_to_tensor(numbers)
    numbers_array_3d = tf.expand_dims(numbers_array, axis=0)
    x_data_transformed = tf.expand_dims(numbers_array_3d, axis=0)

    print("Evaluating ...")

    y = model.predict(x_data_transformed, batch_size=1)
    return {"status": "ok", "result": float(y[0][0])}


if __name__ == '__main__' :
   print ("starting")
   app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
   print ("done")

