import tensorflow as tf


def get_model(hidden_dim, dropout_p, output_dim, learning_rate):

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(hidden_dim),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout_p),
        tf.keras.layers.Dense(output_dim, name='output')
    ])

    if output_dim > 2:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(True)
    else:
        loss = tf.keras.losses.BinaryCrossentropy(True)

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
    )

    return model
