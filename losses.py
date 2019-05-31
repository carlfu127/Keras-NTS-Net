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
    # def rankloss_fn(y_true, y_pred):
    ####y_pred = N*pro*num_cls
    def rank(y_true, y_pred):
        # batch_size, PROPOSAL_NUM, num_cls = K.int_shape(y_pred)
        y_pred = K.reshape(y_pred, (-1, PROPOSAL_NUM, num_cls+1))
        part_prob = -K.log(K.reshape(y_pred[..., :-1], (-1, num_cls))) ##N*pro, *num_cls
        label = K.cast(K.expand_dims(y_true, axis=1), tf.int32)
        targets = K.reshape(K.repeat_elements(label, PROPOSAL_NUM, axis=1), (-1, num_cls))
        # temp = K.softmax(part_logits, axis=-1)
        target_label = tf.argmax(targets, axis=-1, output_type=tf.int32)
        indice = K.stack([tf.range(tf.shape(target_label)[0]), target_label], axis=-1)
        part_pred = tf.gather_nd(part_prob, indice)
        part_pred = K.reshape(part_pred, (-1, PROPOSAL_NUM))
        top_n_prob= y_pred[..., -1]

        loss = 0.
        y_pred = top_n_prob
        y_true = part_pred
        for i in range(PROPOSAL_NUM):
            targets_p = K.cast((y_true > K.expand_dims(y_true[:, i], axis=1)), tf.float32)
            pivot = K.expand_dims(y_pred[:, i], axis=1)
            loss_p = (1 - pivot + y_pred) * targets_p
            loss_p = K.relu(loss_p)
            loss += loss_p
        return K.mean(loss)
    return rank

def concatacc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true, y_pred)
