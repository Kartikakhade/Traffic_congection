from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from traffic_hybrid.config import TrainingConfig


class AttentionPooling(layers.Layer):
    def call(self, inputs, **kwargs):
        sequence_tensor, attention_tensor = inputs
        return tf.reduce_sum(sequence_tensor * attention_tensor, axis=1)

    def compute_output_shape(self, input_shape):
        sequence_shape = input_shape[0]
        return sequence_shape[0], sequence_shape[-1]

    def get_config(self):
        return super().get_config()


class LastStepSlice(layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super().get_config()


def set_random_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def build_bilstm_attention_model(cfg: TrainingConfig, input_shape: tuple[int, int]) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="sequence_input")
    x = inputs

    for index, units in enumerate(cfg.model.hidden_units, start=1):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True),
            name=f"bilstm_{index}",
        )(x)
        if cfg.model.use_layer_normalization:
            x = layers.LayerNormalization(name=f"layer_norm_{index}")(x)
        x = layers.Dropout(cfg.model.dropout, name=f"dropout_{index}")(x)

    attention_scores = layers.Dense(1, activation="tanh", name="attention_score")(x)
    attention_weights = layers.Softmax(axis=1, name="attention_weights")(attention_scores)
    context = AttentionPooling(name="attention_context")([x, attention_weights])

    dense_input = context
    if cfg.model.use_last_step_skip:
        last_step = LastStepSlice(name="last_step")(inputs)
        if cfg.model.use_layer_normalization:
            last_step = layers.LayerNormalization(name="last_step_norm")(last_step)
        dense_input = layers.Concatenate(name="head_concat")([context, last_step])

    dense = layers.Dense(cfg.model.dense_units, activation="relu", name="dense_head")(dense_input)
    if cfg.model.head_dropout > 0.0:
        dense = layers.Dropout(cfg.model.head_dropout, name="head_dropout")(dense)
    output = layers.Dense(1, name="prediction")(dense)

    model = keras.Model(inputs=inputs, outputs=output, name="paper_bilstm_attention")
    optimizer = keras.optimizers.Adam(
        learning_rate=cfg.model.learning_rate,
        clipnorm=cfg.model.gradient_clipnorm,
    )

    compiled_loss = keras.losses.Huber() if cfg.model.loss.lower() == "huber" else "mse"
    model.compile(
        optimizer=optimizer,
        loss=compiled_loss,
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_callbacks(cfg: TrainingConfig, output_dir: Path) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_bilstm_attention.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    if cfg.model.early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.model.early_stopping_patience,
                restore_best_weights=True,
            ),
        )
    if cfg.model.reduce_lr_on_plateau:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.model.reduce_lr_factor,
                patience=cfg.model.reduce_lr_patience,
                min_lr=cfg.model.reduce_lr_min_lr,
            ),
        )

    return callbacks
