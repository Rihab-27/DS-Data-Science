# prediction.py
import numpy as np
from keras.preprocessing import image


def make_prediction(model, image_path, class_mapping):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(result)

    # Get the predicted class name using the class mapping
    predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")
    return predicted_class_name
