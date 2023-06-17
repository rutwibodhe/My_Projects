import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict():
    # Replace this with the path to your image
    image = Image.open('static/images/test_image.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1   #(0 to 255  ==>> -1 to 1)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    idx = np.argmax(prediction)

    if idx == 0:
        return "001.Black_footed_Albatross"
    elif idx == 1:
        return "002.Laysan_Albatross"
    elif idx == 2:
        return "003.Sooty_Albatross"
    elif idx == 3:
        return "004.Groove_billed_Ani"
    elif idx == 4:
        return "005.Crested_Auklet"
    elif idx == 5:
        return "006.Least_Auklet"
    elif idx == 6:
        return "007.Parakeet_Auklet"
    elif idx == 7:
        return "008.Rhinoceros_Auklet"
    elif idx == 8:
        return "009.Brewer_Blackbird"
    elif idx == 9:
        return "010.Red_winged_Blackbird"
    else:
        return "Unknown Bird"

