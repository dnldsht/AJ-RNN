import tensorflow as tf
from config import  Config
import tensorflow_decision_forests as tfdf

import utils
import argparse


tf.config.set_visible_devices([], 'GPU')


def main(config: Config):

    print(f"Training w/ {config.train_data_filename}")
    
    train_dataset, validation_dataset, test_dataset = utils.load_sits_rf(config.seed)

    
    # train_dataset = train_dataset.batch(
    #     config.batch_size, drop_remainder=True)

    # validation_dataset = validation_dataset.batch(
    #     config.batch_size, drop_remainder=True)

    # config.batches = train_dataset.cardinality().numpy()

    print(f"Config {config.__dict__}")


    model = tfdf.keras.RandomForestModel(check_dataset=False)
    model.compile(metrics=["accuracy"])

    callbacks = []

    if config.save_checkpoint:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.checkpoint_path,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_best_only=True)
        )
        print(f"will save weights in {config.checkpoint_path}")

    if config.load_checkpoint:
        print(f"loading weights from {config.checkpoint_path}")
        model.load_weights(config.checkpoint_path)

    
    
    history = model.fit(train_dataset, 
            validation_data=validation_dataset,
            verbose=config.verbose,
            callbacks=callbacks,
            validation_freq=1
            )


    print()
    print("History training")
    utils.print_history(history.history)
    

    if test_dataset is not None:
        # test_dataset = test_dataset.batch(
        #     config.batch_size, drop_remainder=True)
        print()
        print(f"Test Set:")
        model.compile(metrics=["accuracy"])

        history = model.evaluate(test_dataset, verbose=config.verbose, return_dict=True)
        print(history)


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
    parser.add_argument('--GPU', type=str, required=False, default='0', help='GPU to use')

    parser.add_argument('-path', '--checkpoint_path', type=str, required=False, default=None, help='Path of checkpoint model')
    parser.add_argument('-save', '--save_checkpoint', default=False, action='store_true', help='Save model in checkpoint_path')
    parser.add_argument('-load', '--load_checkpoint', default=False, action='store_true', help='Load model from checkpoint')
    parser.add_argument('-small', '--smaller_dataset', default=False, action='store_true', help='Load smaller dataset')
    parser.add_argument('-v', '--verbose', nargs='?', type=int, const=1, default=2, help='Verbose mode')
    parser.add_argument('--seed', type=int, required=True, default=23, help='GPU to use')

    config = parser.parse_args()
    main(config)
