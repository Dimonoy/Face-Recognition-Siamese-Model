import numpy as np
import tensorflow as tf

import cv2


IMAGE_SIZE    = (105, 105)
FACE_DETECTOR = cv2.dnn.readNetFromCaffe(
    'cascades/deploy.prototxt.txt', 
    'cascades/res10_300x300_ssd_iter_140000.caffemodel'
)


class DataPreprocessor:
    """Data workflow for downloading, moving and getting data.
    
    Attributes
    ----------
    anchor_data: np.ndarray
        Anchor images to preprocess.
    positive_data: np.ndarray
        Positive images to preprocess.
    negative_data: np.ndarray
        Negative images to preprocess.
    """
    def __init__(self, anchor_images, positive_images, negative_images) -> None:
        assert anchor_images.shape[0] == positive_images.shape[0] == negative_images.shape[0], \
        "Shapes of all positive, negative and anchor datasets must coincide!"
        
        self.anchor_images   = anchor_images
        self.positive_images = positive_images
        self.negative_images = negative_images
        self.data_length     = anchor_images.shape[0]
    
    @staticmethod
    def augment_image(image: np.ndarray) -> np.ndarray:
        """Augment a single image applying following random transformations:
        
        Brightness adjustment.
        Contrast adjustment.
        Flip from the left to the right.
        Jpeg quality adjustment.
        
        Attributes
        ----------
        image: np.ndarray
            Input image to augment.
        
        Returns
        -------
        np.ndarray: augmented image.
        """
        image = image / 255
        image = tf.image.adjust_brightness(image, delta=0.05)
        image = tf.image.adjust_contrast(image, contrast_factor=0.95)
        image = tf.image.adjust_jpeg_quality(image, jpeg_quality=90)
        image = tf.image.adjust_saturation(image, saturation_factor=0.95)
        image = tf.image.resize(image, IMAGE_SIZE)
        
        return image
        
    def preprocess(self) -> None:
        """Preprocessed inputed images converting them into tf.data.Dataset and applying augmentation.
        """
        self.anchor_images   = tf.data.Dataset.from_tensor_slices(self.anchor_images)
        self.positive_images = tf.data.Dataset.from_tensor_slices(self.positive_images)
        self.negative_images = tf.data.Dataset.from_tensor_slices(self.negative_images)
        
        self.anchor_images   = self.anchor_images.map(self.augment_image, num_parallel_calls=AUTOTUNE)
        self.positive_images = self.positive_images.map(self.augment_image, num_parallel_calls=AUTOTUNE)
        self.negative_images = self.negative_images.map(self.augment_image, num_parallel_calls=AUTOTUNE)
    
    def assemble_images(self) -> tf.data.Dataset:
        """Creates corresponding concatendated image dataset with zipped anchor and positive, zipped anchor and negative images.
        
        Returns
        -------
        tf.data.Dataset: concatenated image dataset.
        """
        zipped_anchor_positive_images = tf.data.Dataset.zip((
            self.anchor_images,
            self.positive_images,
            tf.data.Dataset.from_tensor_slices(tf.ones(self.data_length))
        ))
        zipped_anchor_negative_images = tf.data.Dataset.zip((
            self.anchor_images,
            self.negative_images,
            tf.data.Dataset.from_tensor_slices(tf.zeros(self.data_length))
        ))
        return tf.data.Dataset.concatenate(
            zipped_anchor_positive_images,
            zipped_anchor_negative_images
        )
    
    def train_test_validation_split(self, test_size: float) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """Splits concatenated image dataset on train, test and validation datasets.
        
        Attributes
        ----------
        test_size: float
            The fraction of data will be taken from concatenated image dataset for test split.
        
        Returns
        -------
        tf.data.Dataset: train split dataset.
        tf.data.Dataset: test split dataset.
        """
        assert test_size < 1, "Test fraction size must be less than 1!"
        
        concatenated_images = self.assemble_images()
        concatenated_images = concatenated_images.cache()
        concatenated_images = concatenated_images.shuffle(buffer_size=10000, seed=42)
        
        train_data = concatenated_images.take(round(self.data_length * 2 * (1 - test_size)))
        train_data = train_data.batch(BATCH_SIZE)
        train_data = train_data.prefetch(BATCH_SIZE // 2)
        
        test_data  = concatenated_images.skip(round(self.data_length * 2 * (1 - test_size)))
        test_data  = concatenated_images.take(round(self.data_length * 2 * test_size))
        test_data  = test_data.batch(BATCH_SIZE)
        test_data  = test_data.prefetch(BATCH_SIZE // 2)
        
        return train_data, test_data

def get_faces(image: np.ndarray) -> (np.ndarray, int, int):
    """Apply faces detector to the image.
    
    Attributes
    ----------
    image: np.ndarray
        The image on which faces will be detected.

    Returns
    -------
    np.ndarray: ndarray of faces with specifications.
    int: its width.
    int: its height.
    """
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1,
        (300, 300),
        (104, 117, 123)
    )
    FACE_DETECTOR.setInput(blob)
    faces = FACE_DETECTOR.forward()
    faces_buff = [(confidence, \
                   int(x1 * width), int(y1 * height), \
                   int(x2 * width), int(y2 * height)) \
                 for _, _, confidence, x1, y1, x2, y2 \
                 in faces[0, 0] if confidence > 0.9]
    
    if faces_buff == []:
        return None, None, None, None
    
    _, x1, y1, x2, y2 = max(faces_buff, key=lambda x: x[0])
    return x1, y1, x2, y2
