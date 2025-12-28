import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrainingCallback(Callback):
    """Класс для передачи прогресса обучения обратно в GUI"""
    def __init__(self, update_fn):
        super().__init__()
        self.update_fn = update_fn

    def on_epoch_end(self, epoch, logs=None):
        print(f"DEBUG: End of epoch {epoch+1}. Logs keys: {list(logs.keys())}")
        
        acc = logs.get('accuracy', logs.get('acc', 0.0))
        loss = logs.get('loss', 0.0)
        
        self.update_fn(epoch + 1, acc, loss)

class CnnModel:
    def __init__(self, model_path="models/shape_classifier.keras"):
        self.model = None
        self.model_path = model_path
        self.img_size = (64, 64)
        self.batch_size = 32

    def build_model(self):
        """Создает архитектуру Сверточной нейросети"""
        model = Sequential([
            Input(shape=(64, 64, 1)),
            
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train(self, data_dir, epochs=5, callback_fn=None):
        """Запускает обучение"""
        if self.model is None:
            self.build_model()

        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        if not os.path.exists(data_dir):
            print("ERROR: Папка с данными не найдена!")
            return

        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            subset='training'
        )

        callbacks = []
        if callback_fn:
            callbacks.append(TrainingCallback(callback_fn))

        try:
            self.model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks
            )
            self.model.save(self.model_path)
            print("DEBUG: Модель успешно сохранена.")
        except Exception as e:
            print(f"CRITICAL ERROR during training: {e}")
            
    def predict_image(self, image_path):
        """
        Загружает изображение, обрабатывает его и делает предсказание.
        Возвращает: "Circle" или "Square" и вероятность.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                return "Ошибка", 0.0

        try:
            img = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=self.img_size, 
                color_mode='grayscale'
            )
            
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            img_array /= 255.0
            
            img_array = tf.expand_dims(img_array, 0)
            
            prediction = self.model.predict(img_array)
            probability = prediction[0][0] # Число от 0 до 1
                        
            if probability > 0.5:
                return "Квадрат", probability
            else:
                return "Окружность", 1 - probability

        except Exception as e:
            print(f"Prediction error: {e}")
            return "Ошибка", 0.0