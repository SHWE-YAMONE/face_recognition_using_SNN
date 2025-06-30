from keras.models import load_model
from keras.layers import Layer
import tensorflow as tf

class L1Dist(Layer):
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

model = load_model("siamesemodelv2.keras", custom_objects={"L1Dist": L1Dist})