import numpy as np


def generate_tra_val_tes(samples, k = 40, p_tra = 0.9, seed = None):
    '''
    Def:
        This function splits samples into training, validating and test groups into `k` groups. This grouping is done such that all samples will be
        part of the test set only once (Kfold cross validation). Once the testing samples are selected, the remaining samples are going to be part of the
        trainging and validation set with proportions equal to `p_tra` and 1-`p_tra` correspondilgy. 

    Input: 
        - samples : numpy array (n_samples, ) with the samples IDs to be divided into train, validation and test sets. All samples are going to be included
                    only once in one test set.
        - k       : number of folds. In other words, number of groups.
        - p_tra   : float number [0-1] proportion of training samples. The number of training samples is going to be round down 
                    to favor the number of validation samples.
        - seed    : seed for the random number generator
    
    Output: 
        - tra_val_tes : two level dictionary with:
            - i : integer in string format which corresponds to the index of a K fold
            - tra|val|tes : numpy array (n, ) with the training, validation and test set IDs respectively.
    '''
    from sklearn.model_selection import KFold, train_test_split

    rsg = np.random.RandomState(seed = seed)

    kf = KFold(n_splits = k, shuffle = True, random_state = rsg)
    x  = np.arange(samples.shape[0])
    tra_val_tes = {}

    for i, tra_val_tes_idx in enumerate(kf.split(x)):
        tra_val_tes[str(i)] = {}
        tra_val_idx, tes_idx = tra_val_tes_idx
        tra_idx, val_idx = train_test_split(tra_val_idx, train_size = p_tra,  random_state=rsg, shuffle=True)
        tra_idx.sort()
        val_idx.sort()
        tra_val_tes[str(i)]["tra"], tra_val_tes[str(i)]["val"], tra_val_tes[str(i)]["tes"] = tra_idx, val_idx, tes_idx
    return tra_val_tes


print(generate_tra_val_tes(samples = np.arange(49), k = 5, p_tra = 0.9, seed = 1234))