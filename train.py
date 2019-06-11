import os
from datetime import datetime
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint, \
    TensorBoard, ReduceLROnPlateau, LearningRateScheduler, CSVLogger
from generator import data_generator_wrapper
from config import *
from model import create_attention_model
from losses import *

class ParallelModelCheckpoint(Callback):
    def __init__(self, callback, model):
        super(ParallelModelCheckpoint, self).__init__()
        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

def step_decay(epoch, lr):
    if (epoch+1) == STEPS[0] or (epoch+1) == STEPS[1]:
        lr = lr*0.1
    return lr

if __name__ == '__main__':
    num_gpu = len(gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    save_weights_path = os.path.join(save_weights_path, datetime.now().strftime('%Y%m%d_%H%M%S'))
    batch_size = batch_size*num_gpu

    train_data = open(os.path.join(dataset_dir, 'train.txt')).readlines()
    val_data = open(os.path.join(dataset_dir, 'test.txt')).readlines()

    train_generator = data_generator_wrapper(train_data, batch_size, num_cls, is_train=True)
    val_generator = data_generator_wrapper(val_data, batch_size, num_cls, is_train=False)

    model = create_attention_model(topN=CAT_NUM, PROPOSAL_NUM=PROPOSAL_NUM, num_cls=num_cls)

    checkpoint = ModelCheckpoint(os.path.join(save_weights_path, 'trained_weights_{epoch:04d}.h5'), verbose=1,
                                 monitor='val_loss', mode='auto', save_weights_only=True, save_best_only=True, period=1)
    tensorboard = TensorBoard(log_dir=save_weights_path)
    logs = CSVLogger(filename=os.path.join(save_weights_path, 'training.log'))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.1, patience=10, verbose=1)
    # reduce_lr = LearningRateScheduler(schedule=step_decay, verbose=1)
    if num_gpu > 1:
        training_model = multi_gpu_model(model, gpus=num_gpu)
        checkpoint = ParallelModelCheckpoint(checkpoint, model)
    else:
        training_model = model
    # optimizer = optimizers.RMSprop(lr=initial_learning_rate)
    optimizer = optimizers.SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
    training_model.compile(optimizer=optimizer,
                  loss={'raw': rawloss,
                        'concat': rawloss,
                        'partcls': partloss(PROPOSAL_NUM=PROPOSAL_NUM, num_cls=num_cls),
                        'rank':rankloss(PROPOSAL_NUM=PROPOSAL_NUM, num_cls=num_cls)},
                  metrics={'raw': concatacc,
                      'concat': concatacc})

    training_model.fit_generator(generator=train_generator, steps_per_epoch=len(train_data)//batch_size,
                        epochs=500, verbose=1, callbacks=[checkpoint, tensorboard, logs, reduce_lr],
                        validation_data=val_generator, validation_steps=len(val_data)//batch_size,
                        use_multiprocessing=True, workers=4)
