from kivy.app import App
from kivy.logger import Logger
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock, mainthread
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture

import threading
import numpy as np
import tensorflow as tf
from libs import DataPreprocessor, get_faces

import cv2
import pathlib


class FaceRecognitionApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = pathlib.Path('data/positive')
        self.model     = tf.lite.Interpreter(model_path='fr_model.tflite')
        self.model.allocate_tensors()
        self._set_positive_images()

    def build(self):
        self.web_camera         = Image(size_hint=(1, 0.7))
        self.button_verify      = Button(text="Verification", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text="Verification Unintiated", size_hint=(1, 0.2), font_size=24)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_camera)
        layout.add_widget(self.button_verify)
        layout.add_widget(self.verification_label)

        self.capture = cv2.VideoCapture(cv2.CAP_ANY)
        assert self.capture.isOpened(), "Camera failed to be initialized!"
        Clock.schedule_interval(self.update, 1 / 33)

        return layout

    def update(self, *args):
        _, frame = self.capture.read()
        x1, y1, x2, y2 = get_faces(frame)

        if x1 and y1 and x2 and y2:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        buff          = cv2.flip(frame, 0).tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buff, colorfmt='bgr', bufferfmt='ubyte')

        self.web_camera.texture = image_texture
    
    def _set_positive_images(self):
        positive_images = map(str, (self.data_path).glob('*'))
        positive_images = map(tf.keras.utils.load_img, positive_images)
        positive_images = map(tf.keras.utils.img_to_array, positive_images)
        positive_images = map(DataPreprocessor.augment_image, positive_images)
        positive_images = np.array(list(positive_images))
        self.positive_images = positive_images

    def _apply_model(self, frame):
        outputs         = []
        output_template = self.model.get_output_details()[0]
        inputs_template = self.model.get_input_details()

        for positive_image in self.positive_images:
            self.model.set_tensor(inputs_template[0]['index'], list(np.expand_dims(frame, axis=0)))
            self.model.set_tensor(inputs_template[1]['index'], list(np.expand_dims(positive_image, axis=0)))
            self.model.invoke()

            outputs.append(self.model.get_tensor(output_template['index']))

        return outputs
    
    def verify_thread(self, *args):
        detection_threshold    = 0.5
        verification_threshold = 0.5
        _, frame               = self.capture.read()
        frame                  = np.flip(frame, axis=1)
        x1, y1, x2, y2         = get_faces(frame)

        if x1 and y1 and x2 and y2:
            face = DataPreprocessor.augment_image(frame[y1:y2, x1:x2])
            outputs = self._apply_model(face)

            detection    = np.sum(np.array(outputs) > detection_threshold)
            verification = detection / len(outputs)
            verification = verification > verification_threshold

            self._update_verification_label('Verified' if verification else 'Unverified')


    def verify(self, *args):
        self._update_verification_label()
        threading.Thread(target=self.verify_thread, args=(args)).start()
    
    @mainthread
    def _update_verification_label(self, text=None):
        self.verification_label.text = 'Processing...' if text is None else text


if __name__ == '__main__':
    FaceRecognitionApp().run()
