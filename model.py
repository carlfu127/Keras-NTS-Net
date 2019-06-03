import tensorflow as tf
import keras
from keras.layers import *
from keras.callbacks import *
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from anchors import generate_default_anchor_maps
from config import *

class CropImage(keras.engine.Layer):
    def __init__(self, nbox, pad_side=224, img_size=224, **kwargs):
        self.nbox = nbox
        self.w = img_size
        self.h = img_size
        self.pad_side = pad_side
        super(CropImage, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        image_pad = inputs[0]
        regions = inputs[1]
        regions = K.cast(regions, tf.float32)
        regions = K.reshape(regions, (-1, 4))
        regions_normalized = regions / K.cast(K.int_shape(image_pad)[1], tf.float32)
        box_ind = tf.range(K.shape(image_pad)[0])
        box_ind = K.repeat_elements(box_ind, self.nbox, axis=0)
        self.part_images = tf.image.crop_and_resize(image_pad, regions_normalized, box_ind, (self.w, self.h))
        return self.part_images

    def compute_output_shape(self, input_shape):
        return (None,) + K.int_shape(self.part_images)[1:]

class Proposals(keras.engine.Layer):
    def __init__(self, PROPOSAL_NUM, pad_side=224, **kwargs):
        self.PROPOSAL_NUM = PROPOSAL_NUM
        self.pad_side = pad_side
        edge_anchors = generate_default_anchor_maps()
        self.anchors = (edge_anchors + pad_side).astype(np.int32).astype(np.float32)
        super(Proposals, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        rpn_score = inputs
        indices = K.map_fn(lambda x: tf.image.non_max_suppression(self.anchors, K.cast(x, tf.float32),
                                          self.PROPOSAL_NUM, iou_threshold=0.25, name='nms'), rpn_score, dtype=tf.int32)
        proposals = K.map_fn(lambda x: K.gather(self.anchors, K.cast(x, tf.int32)), indices, dtype=tf.float32)
        scores = K.map_fn(lambda x: K.gather(x[0], K.cast(x[1], tf.int32)), (rpn_score, indices), dtype=tf.float32)
        scores = K.expand_dims(scores, axis=-1)
        results = K.concatenate([proposals, scores], axis=-1)
        self.results = results
        return self.results

    def compute_output_shape(self, input_shape):
        return (None,) + K.int_shape(self.results)[1:]

def res50(num_cls):
    inputs = Input(shape=(None, None, 3))
    base_model = ResNet50(include_top=False, input_tensor=inputs, weights='imagenet', pooling='avg')
    feature1 = base_model.get_layer("activation_49").output
    feature2 = base_model.output
    feature2 = Dropout(rate=0.5)(feature2)
    resnet_out = Dense(num_cls, activation='softmax',
                       kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(weight_decay),
                       bias_regularizer=regularizers.l2(weight_decay))(feature2)
    return keras.models.Model(inputs, [resnet_out, feature1, feature2], name='raw')

def proposalnet():
    inputs = Input(shape=(None, None, 2048))
    down1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                   kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(weight_decay),
                   bias_regularizer=regularizers.l2(weight_decay))(inputs)
    d1 = ReLU()(down1)
    down2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(weight_decay),
                   bias_regularizer=regularizers.l2(weight_decay))(d1)
    d2 = ReLU()(down2)
    down3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(weight_decay),
                   bias_regularizer=regularizers.l2(weight_decay))(d2)
    d3 = ReLU()(down3)

    t1 = Reshape((-1,))(Conv2D(filters=6, kernel_size=1, strides=1,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=regularizers.l2(weight_decay),
                               bias_regularizer=regularizers.l2(weight_decay))(d1))
    t2 = Reshape((-1,))(Conv2D(filters=6, kernel_size=1, strides=1,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=regularizers.l2(weight_decay),
                               bias_regularizer=regularizers.l2(weight_decay))(d2))
    t3 = Reshape((-1,))(Conv2D(filters=9, kernel_size=1, strides=1,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=regularizers.l2(weight_decay),
                               bias_regularizer=regularizers.l2(weight_decay))(d3))

    t = Concatenate(axis=-1)([t1,t2,t3])
    return keras.models.Model(inputs, t)

def concatnet(num_cls, topN):
    inputs = Input(shape=(2048*(topN+1),))
    out = Dense(num_cls, activation='softmax',
                kernel_initializer='he_uniform',
                kernel_regularizer=regularizers.l2(weight_decay),
                bias_regularizer=regularizers.l2(weight_decay))(inputs)
    return keras.models.Model(inputs, out, name='concat')

def partclsnet(num_cls):
    inputs = Input(shape=(2048,))
    out = Dense(num_cls, activation='softmax',
                kernel_initializer='he_uniform',
                kernel_regularizer=regularizers.l2(weight_decay),
                bias_regularizer=regularizers.l2(weight_decay))(inputs)
    return keras.models.Model(inputs, out)

def create_attention_model(topN, PROPOSAL_NUM, num_cls, pad_side=224):
    input_image = Input(shape=(448, 448, 3))

    pretrained_model = res50(num_cls)
    proposal_net = proposalnet()
    concat_net = concatnet(num_cls, topN)
    partcls_net = partclsnet(num_cls)

    resnet_out, rpn_feature, feature = pretrained_model(input_image)
    rpn_feature = Lambda(lambda x: tf.stop_gradient(x))(rpn_feature)
    rpn_score = proposal_net(rpn_feature)

    image_pad = Lambda(lambda x: K.spatial_2d_padding(x,
                padding=((pad_side, pad_side), (pad_side, pad_side))))(input_image)

    top_n_cdds = Proposals(PROPOSAL_NUM=PROPOSAL_NUM, pad_side=pad_side)(rpn_score)
    selected_regions = Lambda(lambda x: x[..., :-1])(top_n_cdds)
    selected_scores = Lambda(lambda x: x[..., -1])(top_n_cdds)
    part_imgs = CropImage(nbox=PROPOSAL_NUM)([image_pad, selected_regions])
    part_imgs = Lambda(lambda x: tf.stop_gradient(x))(part_imgs)
    _, _, part_features = pretrained_model(part_imgs)
    part_feature = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM, 2048)))(part_features)
    part_feature = Lambda(lambda x: x[:, :topN, :])(part_feature)
    part_feature = Reshape((topN*2048,))(part_feature)

    concat_out = Concatenate(axis=1)([part_feature, feature])
    concat_logits = concat_net(concat_out)
    raw_prob = resnet_out
    part_prob = partcls_net(part_features)
    part_prob = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM, num_cls)))(part_prob)
    selected_scores = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM, 1)))(selected_scores)
    concat_rank = Concatenate(axis=-1)([part_prob, selected_scores])
    concat_rank = Reshape((PROPOSAL_NUM*(num_cls+1),), name='rank')(concat_rank)
    part_prob2 = Reshape((PROPOSAL_NUM*num_cls,), name='partcls')(part_prob)
    return keras.models.Model(input_image, [raw_prob, concat_logits, part_prob2, concat_rank])
