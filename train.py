import tensorflow as tf
from ajrnn import AJRNN, LighAJRNN, Config

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import utils
import argparse
import json
import time
import os


tf.config.set_visible_devices([], 'GPU')

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset):
        super().__init__()
        self.test_dataset = test_dataset
        self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

    def on_epoch_end(self, epoch, logs=None):

        res = self.model.evaluate(self.test_dataset, verbose=0, return_dict=True)
        for k,v in res.items():
            logs['test_'+k] = v


def main(config: Config):
    light_ajrnn = config.light_ajrnn

    print(f"Training w/ {config.train_data_filename}")

    wandb.init(
    # set the wandb project where this run will be logged
    project="light-ajrnn" if light_ajrnn else "ajrnn",

    # track hyperparameters and run metadata with wandb.config
    config=config.__dict__,
)
    
    train_dataset, val_dataset, test_dataset, num_classes, num_steps, num_bands = utils.load(config.train_data_filename, config.test_data_filename, config.smaller_dataset, config.seed, light_ajrnn)

    config.num_steps = num_steps
    config.input_dimension_size = num_bands
    config.class_num = num_classes

    train_dataset = train_dataset.batch(
        config.batch_size, drop_remainder=True)

    validation_dataset = val_dataset.batch(
        config.batch_size, drop_remainder=True)

    test_dataset = test_dataset.batch(
            config.batch_size, drop_remainder=True)

    config.batches = train_dataset.cardinality().numpy()

    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)

    
    utils.dump_json(f"{config.results_path}/config.json", config.__dict__)

    model = LighAJRNN(config) if light_ajrnn else AJRNN(config)
    model.compile()
    model.summary()


    model_file = f"{config.results_path}/model/weights"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, save_weights_only=True, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='max')
    logger = tf.keras.callbacks.CSVLogger(f"{config.results_path}/trainlog.csv", separator=',', append=False)
    wandb_logger = WandbMetricsLogger()

    #test_callback = TestCallback(test_dataset)
    # checkpoint, early_stop,
    callbacks = [
        # test_callback,
        wandb_logger,
        checkpoint,
        early_stop,
        logger
    ]
    
    start_train_time = time.time()
    
    history = model.fit(train_dataset, 
            epochs=config.epoch,
            validation_data=validation_dataset,
            verbose=config.verbose,
            callbacks=callbacks,
            validation_freq=1)
    h = history.history

    train_time = round(time.time()-start_train_time, 2)
    

    if test_dataset is not None:
        print()
        print(f"Test ")        
        model.load_weights(model_file)
        
        start_test_time = time.time()
        test_accuracy = model.evaluate(test_dataset, verbose=config.verbose)
        test_time = round(time.time()-start_test_time, 2)
    
    overview = {
        'train_accuracy': h['accuracy'][-1],
        'val_accuracy': max(h['val_accuracy']),
        'test_accuracy': test_accuracy,
        'best_val_epoch': h['val_accuracy'].index(max(h['val_accuracy'])) + 1,
        'train_time': train_time,
        'test_time': test_time
    }
    overview_keys = ['test_accuracy', 'train_time', 'test_time', 'best_val_epoch']
    overview = {k: overview[k] for k in overview_keys if k in overview}
    wandb.log(overview)
    wandb.finish()
    utils.dump_json(f"{config.results_path}/overview.json", overview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--lamda_D', type=float, required=True,help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--G_epoch', type=int, required=True, help='frequency of updating AJRNN in an adversarial training epoch')

    parser.add_argument('--train_data_filename', type=str, required=False, default="SITS")
    parser.add_argument('--test_data_filename', type=str, required=False, default=None)

    parser.add_argument('--layer_num', type=int, required=False, default=1, help='number of layers of AJRNN')
    parser.add_argument('--hidden_size', type=int, required=False, default=100, help='number of hidden units of AJRNN')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3)
    parser.add_argument('--cell_type', type=str, required=False, default='GRU', help='should be "GRU" or "LSTM" ')
    parser.add_argument('--lamda', type=float, required=False, default=1, help='coefficient that balances the prediction loss')
    parser.add_argument('--D_epoch', type=int, required=False, default=1, help='frequency of updating dicriminator in an adversarial training epoch')
    parser.add_argument('--dropout', type=float, default=0, help="Dropout for rnn cell")
    parser.add_argument('--GPU', type=str, required=False, default='0', help='GPU to use')
    parser.add_argument('--reg_loss', default=False, action='store_true', help='Add regularization loss')
    parser.add_argument('--seed', type=int, required=True, default=23, help='GPU to use')
    parser.add_argument('--light_ajrnn', default=False, action='store_true', help='Use light AJRNN')

    parser.add_argument('-results', '--results_path', type=str, required=True, default=None, help='Path of results')
    parser.add_argument('-small', '--smaller_dataset', default=False, action='store_true', help='Load smaller dataset')
    
    parser.add_argument('-v', '--verbose', nargs='?', type=int, const=1, default=2, help='Verbose mode')

    config = parser.parse_args()
    main(config)
