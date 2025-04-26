import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import image_dataset_from_directory

# --- Step 1: Localization Network ---
class LocalizationNetwork(tf.keras.Model): # [cite: 2]
    def __init__(self):
        super(LocalizationNetwork, self).__init__()
        self.conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') # [cite: 2]
        self.pool1 = layers.MaxPooling2D((2, 2)) # [cite: 2]
        self.conv2 = layers.Conv2D(10, (3, 3), activation='relu', padding='same') # [cite: 2]
        self.pool2 = layers.MaxPooling2D((2, 2)) # [cite: 2]
        self.flatten = layers.Flatten() # [cite: 2]
        self.fc1 = layers.Dense(32, activation='relu') # [cite: 3]
        self.fc2 = layers.Dense( # [cite: 3]
            6,
            activation=None,
            kernel_initializer='zeros',
            bias_initializer=tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0])  # Identity transform
        )

    def call(self, x):
        x = self.pool1(self.conv1(x)) # [cite: 3]
        x = self.pool2(self.conv2(x)) # [cite: 4]
        x = self.flatten(x) # [cite: 4]
        x = self.fc1(x) # [cite: 4]
        theta = self.fc2(x) # [cite: 4]
        return theta # [cite: 4]


# --- Step 2: Grid Generator and Bilinear Sampler ---
def affine_grid_generator(theta, input_size): #
    """Generates a sampling grid for the affine transform."""
    # Explicitly cast shape components to int32
    num_batch = tf.cast(input_size[0], tf.int32)
    H = tf.cast(input_size[1], tf.int32)
    W = tf.cast(input_size[2], tf.int32)

    theta = tf.reshape(theta, [-1, 2, 3]) # (N, 2, 3)

    # Normalized grid coordinates
    x = tf.linspace(-1.0, 1.0, W) # W should be int32
    y = tf.linspace(-1.0, 1.0, H) # H should be int32
    x_t, y_t = tf.meshgrid(x, y) # Shapes (H, W), (H, W)

    ones = tf.ones_like(x_t) # (H, W)
    sampling_grid = tf.stack([x_t, y_t, ones], axis=2) # (H, W, 3)

    # Flatten grid
    sampling_grid = tf.reshape(sampling_grid, [1, -1, 3])  # (1, H*W, 3)

    # Repeat grid num_batch times
    sampling_grid = tf.tile(sampling_grid, [num_batch, 1, 1])  # (N, H*W, 3)

    # Transform grid
    # theta shape: (N, 2, 3)
    # sampling_grid shape: (N, H*W, 3)
    # Need to transpose sampling_grid for matmul: (N, 3, H*W)
    grid = tf.matmul(theta, sampling_grid, transpose_b=True)  # (N, 2, H*W)

    # Transpose to (N, H*W, 2) for sampler compatibility
    grid = tf.transpose(grid, [0, 2, 1])

    # Reshape grid back to (N, H, W, 2)
    # Ensure all components of the target shape are int32
    target_shape = tf.stack([num_batch, H, W, 2])
    grid = tf.reshape(grid, target_shape)

    return grid #


def bilinear_sampler(img, grid): # [cite: 6] Modified to remove unused output_size
    """Performs bilinear sampling of the input images according to the normalized grid."""
    B = tf.shape(img)[0] # [cite: 6]
    H = tf.cast(tf.shape(img)[1], tf.float32) # [cite: 6]
    W = tf.cast(tf.shape(img)[2], tf.float32) # [cite: 6]
    C = tf.shape(img)[3] # [cite: 6]

    x_s = grid[..., 0] # [cite: 6]
    y_s = grid[..., 1] # [cite: 6]

    # Scale normalized coordinates to image size
    x = ((x_s + 1.0) * 0.5) * (W - 1.0) # [cite: 6]
    y = ((y_s + 1.0) * 0.5) * (H - 1.0) # [cite: 6]

    x0 = tf.cast(tf.floor(x), tf.int32) # [cite: 7]
    x1 = x0 + 1 # [cite: 7]
    y0 = tf.cast(tf.floor(y), tf.int32) # [cite: 7]
    y1 = y0 + 1 # [cite: 7]

    x0 = tf.clip_by_value(x0, 0, tf.cast(W - 1, tf.int32)) # [cite: 7]
    x1 = tf.clip_by_value(x1, 0, tf.cast(W - 1, tf.int32)) # [cite: 7]
    y0 = tf.clip_by_value(y0, 0, tf.cast(H - 1, tf.int32)) # [cite: 7]
    y1 = tf.clip_by_value(y1, 0, tf.cast(H - 1, tf.int32)) # [cite: 7]

    # Gather pixel values using tf.gather_nd
    batch_indices = tf.tile(tf.reshape(tf.range(B), [B, 1, 1]), [1, tf.cast(H, tf.int32), tf.cast(W, tf.int32)]) # [cite: 8] Explicitly cast H, W

    def gather_pixels(y_indices, x_indices):
        indices = tf.stack([batch_indices, y_indices, x_indices], axis=-1) # [cite: 8]
        return tf.gather_nd(img, indices) # [cite: 8]

    I00 = gather_pixels(y0, x0) # [cite: 8]
    I01 = gather_pixels(y1, x0) # [cite: 8]
    I10 = gather_pixels(y0, x1) # [cite: 8]
    I11 = gather_pixels(y1, x1) # [cite: 8]

    # Interpolation weights
    x = tf.cast(x, tf.float32) # [cite: 8]
    y = tf.cast(y, tf.float32) # [cite: 8]
    x0_f = tf.cast(x0, tf.float32) # [cite: 9]
    y0_f = tf.cast(y0, tf.float32) # [cite: 9]
    x1_f = tf.cast(x1, tf.float32) # Calculate x1_f needed for weights
    y1_f = tf.cast(y1, tf.float32) # Calculate y1_f needed for weights

    wa = (x1_f - x) * (y1_f - y) # [cite: 9]
    wb = (x1_f - x) * (y - y0_f) # [cite: 9]
    wc = (x - x0_f) * (y1_f - y) # [cite: 9]
    wd = (x - x0_f) * (y - y0_f) # [cite: 9]

    wa = tf.expand_dims(wa, -1) # [cite: 9]
    wb = tf.expand_dims(wb, -1) # [cite: 9]
    wc = tf.expand_dims(wc, -1) # [cite: 9]
    wd = tf.expand_dims(wd, -1) # [cite: 9]

    out = wa * I00 + wb * I01 + wc * I10 + wd * I11 # [cite: 9, 10]
    return out # [cite: 10]


# --- Step 3: STN Layer ---
class STN(tf.keras.layers.Layer): # [cite: 10]
    def __init__(self):
        super(STN, self).__init__()
        self.localization_net = LocalizationNetwork() # [cite: 10]

    def call(self, x):
        theta = self.localization_net(x) # [cite: 10]
        input_shape = tf.shape(x) # [cite: 10] # Pass dynamic shape
        grid = affine_grid_generator(theta, input_shape) # [cite: 10]
        sampled = bilinear_sampler(x, grid) # [cite: 11] Removed output_size argument
        return sampled # [cite: 11]


# --- Step 4: Full Emotion Recognition Model ---
def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES): # [cite: 11]
    inputs = Input(shape=input_shape) # [cite: 11]
    x = STN()(inputs) # [cite: 11]

    # Feature extraction
    x = layers.Conv2D(10, (3, 3), activation='relu', padding='same')(x) # [cite: 11]
    x = layers.Conv2D(10, (3, 3), activation='relu', padding='same')(x) # [cite: 11]
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) # [cite: 11]

    x = layers.Conv2D(10, (3, 3), activation='relu', padding='same')(x) # [cite: 11]
    x = layers.Conv2D(10, (3, 3), activation='relu', padding='same')(x) # [cite: 11]
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) # [cite: 12]

    x = layers.Flatten()(x) # [cite: 12]
    x = layers.Dropout(0.5)(x) # [cite: 12]
    x = layers.Dense(50, activation='relu')(x) # [cite: 12]
    output = layers.Dense(num_classes, activation='softmax')(x) # [cite: 12]

    return models.Model(inputs, output) # [cite: 12]