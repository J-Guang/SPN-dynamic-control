"""Pure-TensorFlow Adam / AdamW optimizer.

Implemented with tf.Module + tf.Variable slots so it is fully Keras-free (no
tf.keras.optimizers) and trackable by tf.train.Checkpoint. Used inside the
custom BSDE training loop via ``apply_gradients``.
"""
from __future__ import annotations

import tensorflow as tf


class AdamW(tf.Module):
    def __init__(self, variables, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, clipnorm=0.0, name="adamw"):
        super().__init__(name=name)
        variables = list(variables)
        # Match the optimizer state to the network dtype (float32 or float64), so
        # bias-correction and the lr update stay in one dtype.
        self._dtype = variables[0].dtype if variables else tf.float32
        self.lr = tf.Variable(lr, trainable=False, dtype=self._dtype, name="lr")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.clipnorm = float(clipnorm)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name="step")
        self._m = [tf.Variable(tf.zeros_like(v), trainable=False, name="m")
                   for v in variables]
        self._v = [tf.Variable(tf.zeros_like(v), trainable=False, name="v")
                   for v in variables]

    def set_lr(self, lr: float) -> None:
        self.lr.assign(lr)

    @tf.function(reduce_retracing=True)
    def apply_gradients(self, grads_and_vars):
        grads = [g for g, _ in grads_and_vars]
        variables = [v for _, v in grads_and_vars]
        if self.clipnorm > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
        self.step.assign_add(1)
        t = tf.cast(self.step, self._dtype)
        beta1 = tf.constant(self.beta1, self._dtype)
        beta2 = tf.constant(self.beta2, self._dtype)
        bc1 = 1.0 - tf.pow(beta1, t)
        bc2 = 1.0 - tf.pow(beta2, t)
        for g, v, m_s, v_s in zip(grads, variables, self._m, self._v):
            if g is None:
                continue
            g = tf.cast(g, v.dtype)
            m_s.assign(self.beta1 * m_s + (1.0 - self.beta1) * g)
            v_s.assign(self.beta2 * v_s + (1.0 - self.beta2) * tf.square(g))
            m_hat = m_s / bc1
            v_hat = v_s / bc2
            update = m_hat / (tf.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0.0:
                update = update + self.weight_decay * v
            v.assign_sub(self.lr * update)
