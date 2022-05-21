from typing import Union

import tensorflow as tf


def smelu(x: Union[list, tf.Tensor], beta: float = 1.):
    """
    Implementation of the Smooth ReLU (SmeLU) activation function developed
    in Real World Large Scale Recommendation Systems Reproducibility and Smooth Activations
    
    Parameters
    ----------
    x: List, Numpy or Tensorflow tensor
    beta: float
        Half-width of a symmetric transition region around x = 0
        
    Examples
    --------
    >>> import tensorflow as tf
    >>> from tf_smelu import smelu
    
    >>> x = tf.range(-6, 6, 1, dtype=float)
    # <tf.Tensor: numpy=array([-6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.], dtype=float32)>
    >>> smelu(x, beta=0.1)
    # <tf.Tensor: numpy=array([0., 0., 0., 0., 0., 0., 0.025, 1., 2., 3., 4., 5.], dtype=float32)>
        
    See Also
    --------
    - https://arxiv.org/pdf/2202.06499.pdf
    """
    x = tf.convert_to_tensor(x)
    return tf.where(tf.math.abs(x) <= beta, ((x + beta) ** 2) / (4 * beta), tf.nn.relu(x))
