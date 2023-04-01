import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt

# x_train = 1
# x_test = 1

def model_builder(hp):
    model = Sequential()

    # input layer 
    model.add(Dense(units=x.shape[0], input_shape=(x.shape[1],)))

    # let the model decide how many layers it wants to have
    for i in range(hp.Int("num_layers", min_value=2, max_value=5, step=1)):
        model.add(Dropout(hp.Float("dropout_" + str(i), min_value=0, max_value=0.7, step=0.1)))
        model.add(
            Dense(
                units=hp.Int("layer_" + str(i), min_value=16, max_value=1024, step=64),
                activation=hp.Choice("act_" + str(i), ["relu", "sigmoid"]),
            )
        )

    # output shape of the model the same as the number of features
    model.add(Dense(y.shape[1]), activation=hp.Choice("output_layer_act", ["relu", "sigmoid", "softmax"]))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def hp_search(x_train, y_train):
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

    tuner = kt.BayesianOptimization(
        model_builder,
        objective="val_accuracy",
        directory=r"../logs",
        project_name="BayOpt_v1.01",
    )

    tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callback=callback)

    return tuner.get_best_models()[0]
