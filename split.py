import numpy as np
from sklearn.utils import shuffle

def getData(objs, labels, data, k):
    objID_col = labels[:,3]
    tot_data = []
    for obj in objs:
        idx = np.where(objID_col == obj)
        tot_data.append( data[idx] )
    tot_data = np.concatenate(tot_data,axis=0)
    return tot_data, np.ones(tot_data.shape[0])*k

def getCL2Objs(labels):
    # recupero tutte le classi
    clID_col = labels[:,2]

    # set di classi
    clID = np.unique(clID_col)
    hashClID2obj = {}
    # per ogni classe
    for val in clID:
        # recupero gli oggetti con classe val
        idx = np.where(clID_col == val)
        t_labels = labels[idx]
        # recupero gli objectID della classe val
        hashClID2obj[val] = np.unique( t_labels[:,3] )
    return hashClID2obj

data =np.load("data.npy")
labels = np.load("labels.npy")
hashClID2obj = getCL2Objs(labels)
train_perc = .5
train_valid = .2

tot_train_x = []
tot_train_y = []
tot_valid_x = []
tot_valid_y = []
tot_test_x = []
tot_test_y = []

for k in hashClID2obj.keys():
    # recupero gli oggetti della classe k
    objIds = hashClID2obj[k]
    objIds = shuffle(objIds)


    # divido gli objectID in train, valid e test
    limit_train = int(len(objIds)* train_perc )
    limit_valid = limit_train + int(len(objIds)* train_valid)

    train_obj = objIds[0:limit_train]
    valid_obj = objIds[limit_train:limit_valid]
    test_obj = objIds[limit_valid::]



    train_x, train_y = getData(train_obj, labels, data, k)
    tot_train_x.append(train_x)
    tot_train_y.append(train_y)

    valid_x, valid_y = getData(valid_obj, labels, data, k)
    tot_valid_x.append(valid_x)
    tot_valid_y.append(valid_y)

    test_x, test_y = getData(test_obj, labels, data, k)
    tot_test_x.append(test_x)
    tot_test_y.append(test_y)

np.save("x_train.npy", np.concatenate(tot_train_x,axis=0))
np.save("y_train.npy", np.concatenate(tot_train_y,axis=0))
np.save("x_valid.npy", np.concatenate(tot_valid_x,axis=0))
np.save("y_valid.npy", np.concatenate(tot_valid_y,axis=0))
np.save("x_test.npy", np.concatenate(tot_test_x,axis=0))
np.save("y_test.npy", np.concatenate(tot_test_y,axis=0))

# Per evitare l'auto correlazione spaziale, devo dividere i dati in train, valid e test facendo in modo che nei vari dataset
# ci sia la stessa distribuzione di classi pero con oggetti diversi (non mischiare objectID nei vari dataset).