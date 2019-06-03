import tensorflow as tf
import keras
from keras import backend as K

def rawloss(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true, y_pred))

def partloss(PROPOSAL_NUM, num_cls):
    def part(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, PROPOSAL_NUM, num_cls))
        y_true = K.repeat_elements(K.expand_dims(y_true, axis=1), PROPOSAL_NUM, axis=1)
        return K.mean(K.categorical_crossentropy(y_true, y_pred))
    return part

def rankloss(PROPOSAL_NUM, num_cls):
    def rank(y_true, y_pred):
        ####y_pred = N*pro*(num_cls+1)
        y_pred = K.reshape(y_pred, (-1, PROPOSAL_NUM, num_cls+1))
        part_prob = -K.log(K.reshape(y_pred[..., :-1], (-1, num_cls))) ##N*pro, *num_cls
        label = K.cast(K.expand_dims(y_true, axis=1), tf.int32)
        label = K.reshape(K.repeat_elements(label, PROPOSAL_NUM, axis=1), (-1, num_cls)) ##N*pro, *num_cls
        target_label = tf.argmax(label, axis=-1, output_type=tf.int32)  ##N*pro
        indice = K.stack([tf.range(tf.shape(target_label)[0]), target_label], axis=-1)
        part_pred = tf.gather_nd(part_prob, indice)
        part_pred = K.reshape(part_pred, (-1, PROPOSAL_NUM)) ##N*pro
        top_n_prob= y_pred[..., -1] ##N*pro

        loss = 0.
        score = top_n_prob
        targets = part_pred
        for i in range(PROPOSAL_NUM):
            targets_p = K.cast((targets > K.expand_dims(targets[:, i], axis=1)), tf.float32)
            pivot = K.expand_dims(score[:, i], axis=1)
            loss_p = (1 - pivot + score) * targets_p
            loss_p = K.relu(loss_p)
            loss += loss_p
        return K.mean(loss)
    return rank

def concatacc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true, y_pred)
