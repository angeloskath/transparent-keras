Transparent Keras
=================

Transparent aims to provide a very simple way to look under the hood during
training of Keras models by defining an extra set of outputs that will be
returned by `train_on_batch` or `evaluate_on_batch`.

Example
-------

.. code:: python

    from keras.layers import Activation, Dense, Dropout, Input
    import numpy as np

    from transparent_keras import TransparentModel

    x0 = Input(shape=(10,))
    x = Dense(10, activation="relu")(x0)
    x = Dropout(0.5)(x)
    y_extra = x = Dense(10)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(1)(x)

    m = TransparentModel(inputs=[x0], outputs=[y], observed_tensors=[y_extra])
    m.compile(optimizer="sgd", loss="mse")

    x_random = np.random.rand(128, 10)
    y_random = np.random.rand(128, 1)
    loss, y_extra = m.train_on_batch(x_random, y_random)
