# main.py
from model import build_model, train_model
from prediction import make_prediction
from keras.preprocessing.image import ImageDataGenerator


def main():
    class_mapping = {0: 'hyundai', 1: 'lexus', 2: 'mazda', 3: 'mercedes', 4: 'opel', 5: 'skoda', 6: 'toyota',
                     7: 'volkswagen'}

    model = None  # Initialize the model outside the loop

    while True:
        print("\nOptions:")
        print("1. Train the model")
        print("2. Make a prediction")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            # Load and preprocess the dataset
            train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=32,
                                                             class_mode='categorical')
            test_set = test_datagen.flow_from_directory('dataset/test', target_size=(64, 64), batch_size=32,
                                                        class_mode='categorical')

            # Build and train the model
            model = build_model()
            model = train_model(model, training_set, test_set)

        elif choice == '2':
            if model is None:
                print("Error: Model not trained. Please train the model first.")
            else:
                # Make a prediction
                image_path = input("Enter the path to the test image: ")
                prediction = make_prediction(model, image_path, class_mapping)
                print(f"The predicted class is: {prediction}")

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
