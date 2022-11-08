class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    layer_num = 1 # number of layers of AJRNN
    hidden_size = 100
    learning_rate = 1e-3
    cell_type = 'GRU'
    lamda = 1 # coefficient that balances the prediction loss
    D_epoch = 1
    GPU = '0'
    '''User defined'''
    batch_size = None  # batch_size for train
    epoch = None  # epoch for train
    lamda_D = None  # epoch for training of Discriminator
    G_epoch = None  # epoch for training of Generator
    batches = None
    train_data_filename = None
    test_data_filename = None
    results_path = None
    smaller_dataset = False
    verbose = 2
    seed = 23