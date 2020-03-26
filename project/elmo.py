import tensorflow_hub as hub
import numpy as np
import pandas as pd

import tensorflow as tf


elmo = hub.Module("pretrained", trainable=False)
embeddings = elmo(
        ["the cat is on the mat", "what are you doing in evening"],
        signature="default",
        as_dict=True)["elmo"]

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embeddings)


message_embeddings.shape

import ipdb
ipdb.set_trace()

