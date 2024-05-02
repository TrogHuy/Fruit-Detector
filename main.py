from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from Adafruit_IO import MQTTClient
import time
from random import randint
import base64
import io

AIO_USERNAME = "ngtronghuy"
AIO_KEY = "aio_pBbY28Km2kOYy7NgJU0Noq8yMAcF"

client = MQTTClient(AIO_USERNAME, AIO_KEY)
client.connect()

while True:

    image_paths = [
        [
            "test_model_data/apple/apple1.jpg",
            "test_model_data/apple/apple2.jpg",
            "test_model_data/apple/apple3.jpg",
        ],
        [
            "test_model_data/banana/banana1.jpg",
            "test_model_data/banana/banana2.jpg",
            "test_model_data/banana/banana3.png",
        ],
        [
            "test_model_data/dragon fruit/dragon fruit1.jpg",
            "test_model_data/dragon fruit/dragon fruit2.jpg",
            "test_model_data/dragon fruit/dragon fruit3.jpg",
        ],
        [
            "test_model_data/durian/durian1.jpg",
            "test_model_data/durian/durian2.jpg",
            "test_model_data/durian/durian3.jpg",
        ],
        [
            "test_model_data/grape/grape1.jpg",
            "test_model_data/grape/grape2.jpg",
            "test_model_data/grape/grape3.jpg",
        ],
        [
            "test_model_data/orange/orange1.jpg",
            "test_model_data/orange/orange2.jpg",
            "test_model_data/orange/orange3.jpg",
        ],
        [
            "test_model_data/pineapple/pineapple1.jpg",
            "test_model_data/pineapple/pineapple2.jpg",
            "test_model_data/pineapple/pineapple3.jpg",
        ],
        [
            "test_model_data/tomato/tomato1.jpg",
            "test_model_data/tomato/tomato2.jpg",
            "test_model_data/tomato/tomato3.jpg",
        ],
        [
            "test_model_data/no object/no object1.jpg",
            "test_model_data/no object/no object2.jpg",
            "test_model_data/no object/no object3.png",
        ],
    ]

    row = randint(0, 8)
    column = randint(0, 2)

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_paths[row][column]).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2] + "%")

    client.publish("Fruits Detected", class_name[2:])
    client.publish("Accuracy", str(np.round(confidence_score * 100))[:-2])
    client.publish("Image", img_str)

    time.sleep(2)
