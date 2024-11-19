import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, PReLU


# Define a custom attention layer
class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        # Initialize weights for the attention mechanism
        self.attention_weights = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Calculate attention scores
        attention_logits = tf.matmul(inputs, self.attention_weights)
        # Apply softmax to normalize the attention weights
        attention_weights = tf.nn.softmax(attention_logits)
        # Weight the inputs by attention scores
        attended_inputs = inputs * attention_weights
        return attended_inputs


# Define a custom self-attention layer
class SelfAttentionLayer(layers.Layer):
    def __init__(self, d_model):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model

    def build(self, input_shape):
        # Initialize weights for queries, keys, and values
        self.query_weights = self.add_weight(
            shape=(input_shape[-1], self.d_model),
            initializer='random_normal',
            trainable=True
        )
        self.key_weights = self.add_weight(
            shape=(input_shape[-1], self.d_model),
            initializer='random_normal',
            trainable=True
        )
        self.value_weights = self.add_weight(
            shape=(input_shape[-1], self.d_model),
            initializer='random_normal',
            trainable=True
        )
        # Define a dense layer for output projection
        self.dense = layers.Dense(input_shape[-1])
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute queries, keys, and values
        queries = tf.matmul(inputs, self.query_weights)
        keys = tf.matmul(inputs, self.key_weights)
        values = tf.matmul(inputs, self.value_weights)

        # Scaled dot-product attention
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        attention_logits = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        attended_values = tf.matmul(attention_weights, values)

        # Output projection
        output = self.dense(attended_values)
        return output


# Main model class for single-cell convolutional autoencoder
class scCAE:
    """
    Single-cell Convolutional Autoencoder (scCAE)

    Parameters:
        input_shape: int
            Shape of the input data (number of features).
        kernel_height1: int
            Height of the kernel for the first convolution layer.
        kernel_height2: int
            Height of the kernel for the second convolution layer.
        stride1: int
            Stride for the first convolution layer.
        stride2: int
            Stride for the second convolution layer.
        filter1: int
            Number of filters in the first convolution layer.
        filter2: int
            Number of filters in the second convolution layer.
        hidden_dim: int
            Dimension of the hidden representation.
        d_model: int
            Dimension for self-attention projection.
        optimizer: tf.keras.optimizers.Optimizer
            Optimizer for model training.
        epochs: int
            Number of training epochs.
        batch_size: int
            Size of each training batch.
        validation_rate: float
            Fraction of data used for validation.
        learning_rate: float
            Learning rate for the optimizer.
    """

    def __init__(self, input_shape, kernel_height1=10, kernel_height2=5, stride1=5, stride2=2, filter1=16, filter2=32,
                 hidden_dim=30, d_model=32, optimizer=tf.keras.optimizers.Adam, epochs=100, batch_size=32,
                 validation_rate=0.2, learning_rate=0.0005):
        # Initialize hyperparameters
        self.kernel_height1 = kernel_height1
        self.kernel_height2 = kernel_height2
        self.stride1 = stride1
        self.stride2 = stride2
        self.filter1 = filter1
        self.filter2 = filter2
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_rate = validation_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        # Create a sequential model
        self.model = Sequential()

        # Add convolutional layers
        self.model.add(Conv2D(self.filter1, (1, self.kernel_height1), strides=(1, self.stride1), padding='same',
                              input_shape=(1, input_shape, 1)))
        self.model.add(BatchNormalization())  # Batch normalization
        self.model.add(PReLU())  # PReLU activation

        self.model.add(Conv2D(self.filter2, (1, self.kernel_height2), strides=(1, self.stride2), padding='same'))
        self.model.add(BatchNormalization())  # Batch normalization
        self.model.add(PReLU())  # PReLU activation

        # Add attention layer
        self.model.add(AttentionLayer())

        # Flatten the output
        self.model.add(Flatten())

        # Add a dense layer for hidden representation
        self.model.add(Dense(units=self.hidden_dim, name='HiddenLayer'))
        self.model.add(BatchNormalization())  # Batch normalization
        self.model.add(PReLU())  # PReLU activation

        # Add dense and reshape layers for reconstruction
        self.model.add(Dense(units=self.filter2 * int(input_shape / (self.stride1 * self.stride2))))
        self.model.add(Reshape((1, int(input_shape / (self.stride1 * self.stride2)), self.filter2)))

        # Add transposed convolutional layers for reconstruction
        self.model.add(
            Conv2DTranspose(self.filter1, (1, self.kernel_height2), strides=(1, self.stride2), padding='same'))
        self.model.add(BatchNormalization())  # Batch normalization
        self.model.add(PReLU())  # PReLU activation

        self.model.add(Conv2DTranspose(1, (1, self.kernel_height1), strides=(1, self.stride1), padding='same'))
        self.model.add(BatchNormalization())  # Batch normalization
        self.model.add(PReLU())  # PReLU activation

        # Print model summary
        self.model.summary()

    def fit(self, data):
        # Reshape input data to match the model's expected shape
        data_train = tf.reshape(data, [-1, 1, data.shape[1], 1])
        print(data.shape)
        print(data_train.shape)
        # Compile the model with mean squared error loss and the specified optimizer
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer(lr=self.learning_rate))

        # Train the model on the reshaped input data
        self.model.fit(data_train, data_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_split=self.validation_rate)

    def make_encoders(self):
        # Create an encoder model
        encoder = Model(inputs=self.model.input, outputs=self.model.get_layer(name='HiddenLayer').output,
                        name="encoder")
        return self.model, encoder

    def extract_feature(self, x):
        # Extract features from the hidden layer
        x = tf.reshape(x, [-1, 1, x.shape[1], 1])
        feature_model = Model(inputs=self.model.input, outputs=self.model.get_layer(name='HiddenLayer').output)
        features = feature_model.predict(x)
        return features
