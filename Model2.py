import joblib
import tensorflow as tf

# Load the scikit-learn model
model = joblib.load('fertilizer_recommendation_model.pkl')

# Define input shape based on your scikit-learn model's input requirements
input_shape = (10,)  # Example input shape with 10 features

# Convert the scikit-learn model to TensorFlow's SavedModel format
# Rebuild the model in TensorFlow
inputs = tf.keras.Input(shape=input_shape)
# Define your TensorFlow model architecture here based on the scikit-learn model's requirements
outputs = tf.keras.layers.Dense(1)(inputs)  # Example with a single output node

# Create a TensorFlow model
tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model if necessary
# tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Save the model
tf.saved_model.save(tf_model, 'tf_saved_model')

# Convert the SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('tf_saved_model')
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
