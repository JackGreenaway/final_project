import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
import keras_tuner as kt


def model_builder(hp):
    model = Sequential()

    # input layer 
    model.add(Dense(units=self.x_train.shape[0], input_shape=(self.x_train.shape[0], self.x_train.shape[1])))

    # let the model decide how many layers it wants to have
    for i in range(hp.Int("num_layers", min_value=2, max_value=5, step=1)):
        model.add(
            Dense(
                units=hp.Int("layer_" + str(i), min_value=16, max_value=512, step=16),
                activation=hp.Choice("act_" + str(i), ["relu", "sigmoid"]),
            )
        )

    # output shape of the model the same as the number of features
    model.add(Dense(self.x_train.shape[0]), activation="softmax")

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def hp_search(x_train, y_train):
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        project_name="hyperband_v1.01",
    )

    tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callback=callback)

    return tuner.get_best_models()[0]
