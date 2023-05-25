#A. Import
import numpy      as np
import pandas     as pd
import os
import sys
import json
from matplotlib  import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import simGL


class CustomDataGen(tf.keras.utils.Sequence):
    '''
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    https://keras.io/api/models/model_training_apis/
    '''
    def __init__(self, x, y, batch_size = 32, x_weights = np.array([]), shuffle = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        if x_weights.shape == (0,):
            self.x_weights = np.ones(x.shape[0])
        else:
            self.x_weights = x_weights
    
    def __len__(self):
        return int(np.ceil(self.x.shape[1] / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            i = np.arange(self.x.shape[0])
            np.random.shuffle(i)
            self.x = self.x[i]
            self.y = self.y[i]
            self.x_weights = self.x_weights[i]

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        batch_x_weights = self.x_weights[low:high]

        return batch_x, batch_y, batch_x_weights


#B. Functions
#B.1
def generate_tra_val_tes(samples, k = 50, p_tra = 0.9, seed = None):
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
        tra_val_tes[str(i)]["tra"], tra_val_tes[str(i)]["val"], tra_val_tes[str(i)]["tes"] = tra_idx.tolist(), val_idx.tolist(), tes_idx.tolist()
    return tra_val_tes

#B.2
def get_tra_val_tes(samples, file = "", overwrite = False, k = 50, p_tra = 0.9, seed = None):
    '''
    Def:
        This function generates all the different sets of training, validating and testing sets such that all samples in the "samples" variable
        are part of the test set once.

        If there is an existing file with such random sets, it is possible to get the sets from the file, generate a new one and overwrite or
        ignore the file and generate a set without storing it in a file.

    Input:
        - samples   : numpy array (n_samples, ) with the samples IDs to be divided into train, validation and test sets. All samples are going to be included
                      only once in one test set.
        - file      : string  with the file name and path where the output dictionary "tra_val_tes" is going to be stored in json format. It is important that
                      the file termination is ".json". If the string is empty, 
        - overwrite : boolean variable that will decide if the already stored file should be ignored, generate a new set and overwrite the file.
        - n_tes     : integer with the test set size. There will be round_up(len(samples)/n_tes) test sets in total.
        - p_tra     : float number [0-1] proportion of training samples to validation samples. The number of training samples is going to be round down 
                      to favor the number of validation samples.
    
    Output: 
        - tra_val_tes : two level dictionary with:
            - i : integer corresponding to the index of a set
            - tra|val|tes : numpy array (n, ) with the training, validation and test set IDs respectively.
    '''
    if file:
        if os.path.isfile(file) and not overwrite:
            with open(file, 'r') as tra_val_tes_file:
                tra_val_tes = json.load(tra_val_tes_file)
        else:
            tra_val_tes = generate_tra_val_tes(samples = samples, k = k, p_tra = p_tra, seed = seed)
            with open(file, 'w') as tra_val_tes_file:
                json.dump(tra_val_tes, tra_val_tes_file)
    else:
        tra_val_tes = generate_tra_val_tes(samples = samples, k = k, p_tra = p_tra, seed = seed)
        
    return tra_val_tes

#B.3
def get_input_simulated_tree(ts, metadata, snp, typ, cov, err):
    '''
    Def :
        Obtain the allele counts, genotype likelihoods, genotype probablities or other for variant sites from a simulated tree structure (ts) 
        that are not singletons or nearly fixed for a subset of individuals indicated in a metadata pandas DataFrame.
    Input :
        - ts       : tree data structure
        - metadata : pandas DataFrame (n_individuals, features) in which for each individual there are indicated both nodes ID in the ts in
                     columns named "node1" and "node2"
        - snp      : float [0-1] denoting the proportion of variants to be included. This can be helpful if the user wants to experiment
                     with the number of snps in the input
        - typ      : str denoting the type of input that should be outputed by the function:
                        - gt     : genotypes in a flat vector for all positions
                        - gl_mix : mixture input that gives the gt of the allele and the second most likely genotype likelihood
                        - gl     : the 3 genotype likelihoods in a flat vector for all positions. In other words 
                                   the first 3 values of the output of this function will correspond to the genotype likelihoods
                                   of the first variant and the next three, to the following variant and so on. 
                        - gp     : similar to gl but for genotype probabilities. Right now genotype probabilities are computed
                                   with a flat prior
                        - gp_mix : similar to gl_mix but with gp
                        - gl_gt  : similar to gt, but the gt represents the genotype with the highest gl
    Output :
        -  allele_counts : numpy array (n_individuals, n_variants) with allele counts per individual.
            - 0 : Ancestral allele homozigous
            - 1 : Heterozygous
            - 2 : Derived allele homozigous 
    '''
    np.random.seed(1234)
    gm = []
    for v in ts.variants(samples = metadata[["node1", "node2"]].to_numpy().reshape(-1).tolist()):
        gl = v.genotypes.sum()
        #the allele is polymorphic and not a singleton, it is biallelic and randombly is sampled (this is used when we want to keep p number of alleles)
        if gl > 1 and gl < metadata.shape[0]-1 and len(v.alleles) < 3 and np.random.binomial(1, snp):
            if typ == "gt":
                gm.append(v.genotypes.reshape(-1, 2).sum(axis = 1).tolist())
            elif typ in ["gl", "gl_mix", "gp", "gp_mix", "gl_gt"]:
                gm.append(v.genotypes.tolist())
    gm = np.array(gm)
    if typ == "gt":
        return gm.T
    elif typ == "gl_mix":
        arc      = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL       = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]

        argsorGL = np.argsort(GL)

        minGLidx = argsorGL[:, :, 0].reshape(-1)#np.argmax(argsorGL == 0, axis = 2).reshape(-1)

        dim1_idx = np.repeat(np.arange(GL.shape[0]), GL.shape[1])
        dim2_idx = np.tile(np.arange(GL.shape[1]), GL.shape[0])
        dim3_idx = argsorGL[:, :, 1].reshape(-1)#np.argmax(np.argsort(GL) == 1, axis = 2).reshape(-1)
        midGLval = GL[dim1_idx, dim2_idx, dim3_idx]

        missing  = ((GL == 0).sum(axis = 2) == 3).reshape(-1)

        minGLidx[missing] = -1#np.random.choice([0, 1, 2]) #-1
        midGLval[missing] = -1#0 #midGLval[missing] = max(midGLval)*2

        GLmix = []
        for i in range(GL.shape[1]):
            GLmix.append(np.dstack((minGLidx[i::GL.shape[1]], midGLval[i::GL.shape[1]])).flatten().tolist())
        GLmix = np.array(GLmix)

        return GLmix
    elif typ == "gp_mix":
        arc      = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL       = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        likelihood = np.exp(-GL)
        prior      = np.array([1/4, 1/2, 1/4])
        GP         = likelihood*prior/(np.sum(likelihood*prior, axis = 2).reshape(GL.shape[:2] + (1,)))

        argsorGP = np.argsort(GP)

        minGPidx = argsorGP[:, :, 2].reshape(-1)

        dim1_idx = np.repeat(np.arange(GL.shape[0]), GL.shape[1])
        dim2_idx = np.tile(np.arange(GL.shape[1]), GL.shape[0])
        midGPval = GP[dim1_idx, dim2_idx, minGPidx]

        missing  = ((GL == 0).sum(axis = 2) == 3).reshape(-1)

        minGPidx[missing] = -1#np.random.choice([0, 1, 2]) #-1
        midGPval[missing] = -1#0 #midGPval[missing] = max(midGPval)*2

        GPmix = []
        for i in range(GL.shape[1]):
            GPmix.append(np.dstack((minGPidx[i::GP.shape[1]], midGPval[i::GP.shape[1]])).flatten().tolist())
        GPmix = np.array(GPmix)

        return GPmix
    elif typ == "gl":
        arc      = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL       = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        return GL.transpose((1, 0, 2)).reshape(-1).reshape(GL.shape[1], GL.shape[0]*3)
    elif typ == "gp":
        arc        = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL         = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        likelihood = np.exp(-GL)
        prior      = np.array([1/4, 1/2, 1/4])
        GP         = likelihood*prior/(np.sum(likelihood*prior, axis = 2).reshape(GL.shape[:2] + (1,)))
        return GP.transpose((1, 0, 2)).reshape(-1).reshape(GP.shape[1], GP.shape[0]*3)
    elif typ == "gl_gt":
        arc      = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL       = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        
        minGLidx = np.argsort(GL)[:, :, 0]
        
        missing  = ((GL == 0).sum(axis = 2) == 3)
        minGLidx[missing] = -1

        return minGLidx.T

def get_input_AADR(metadata, chrom, start, end):
    snp     = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.snp", index_col = None, header = None, names = ["snp", "chr", "gen", "pos", "ref", "alt"])

    snp_idx = np.array([i for i in snp[np.in1d(snp.chr, chrom) & (snp.pos >= start) & (snp.pos < end)].index])
    print()
    geno = []
    if snp_idx.shape[0]:
        with open("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public.eigenstratgeno", "r") as f:
            for i, l in enumerate(f):
                if i >= snp_idx[0] and i <= snp_idx[-1]:
                    geno.append([int(x) for x in l.strip()])
                elif i > snp_idx[-1]:
                    break
                
        geno = np.array(geno)[:, metadata.index.to_numpy()]

        geno[geno == 9] = -1

        return geno.T, snp[np.in1d(snp.chr, chrom) & (snp.pos >= start) & (snp.pos < end)].pos.to_numpy()
    else:
        return None, None


def get_output(pre, metadata):
    '''
    Def :
        Extract the output that the NN will predict from the input.
    Input :
        - pre : string that defines if the output that is extracted has to be:
            - sNNt  : space and time
            - space : only space (latitude and longitude)
            - time  : only time
        - metadata : the pandas DataFrame from which the output will be extracted.
    Output :
        - numpy array with the values of the columns corresponding to the information that 
          has to be extracted.
    '''
    if   pre == "sNNt":
        return metadata[["lat", "lon", "time"]].to_numpy()
    elif pre == "space":
        return metadata[["lat", "lon"]].to_numpy()
    elif pre == "time":
        return metadata[["time"]].to_numpy()
    else:
        sys.exit("Incorrect prediction value {}\n".format(pre))

def carryon(carryon_file):
    if os.path.exists(carryon_file):
        f = open(carryon_file, "r")
        start_batch = int(f.read())+1
        new_file    = False
        print("Batches up to {} already run. Starting from there...\n".format(start_batch-1))
    else:
        start_batch = 0
        new_file    = True
        print("No batches run yet. Starting form first batch...\n")
    return start_batch, new_file

def normalizer(nor, array):
    if nor == "None":
        mean     = np.array([0.])
        variance = np.array([1.])
        layer = tf.keras.layers.Normalization(input_shape=[array.shape[1], ], axis = None, mean = mean, variance = variance)
    elif nor == "Norm0":
        mean     = array.mean(axis = None)
        variance = array.var(axis = None)
        layer = tf.keras.layers.Normalization(input_shape=[array.shape[1], ], axis = None, mean = mean, variance = variance)
    elif nor == "Norm1":
        mean     = array.mean(axis = 0).reshape(1, -1)
        variance = array.var(axis = 0).reshape(1, -1)
        layer = tf.keras.layers.Normalization(input_shape=[array.shape[1], ], axis = -1)
    if nor != "None":
        layer.adapt(array)
    return layer, mean, variance

#B.6
def callbacks(weights_file_name):
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath          = weights_file_name,
        verbose           = 0,
        save_best_only    = True,
        save_weights_only = True,
        monitor           = "val_loss",
        save_freq         = "epoch")

    earlystop=tf.keras.callbacks.EarlyStopping(
        monitor              = "val_loss",
        min_delta            = 0,
        patience             = 100,
        verbose              = 1,
        restore_best_weights = True)
    
    reducelr=tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 20)
    return checkpoint, earlystop, reducelr

#B.7
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis = -1))
#B.8
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis = -1)

#B.9
def tf_atan2(y, x):
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle

#B.10
def tf_haversine(latlon1, latlon2):
    lat1 = latlon1[:, 0]
    lon1 = latlon1[:, 1]
    lat2 = latlon2[:, 0]
    lon2 = latlon2[:, 1]

    REarth = 6371
    lat = tf.abs(lat1 - lat2) * np.pi / 180
    lon = tf.abs(lon1 - lon2) * np.pi / 180
    lat1 = lat1 * np.pi / 180
    lat2 = lat2 * np.pi / 180
    a = tf.sin(lat / 2) * tf.sin(lat / 2) + tf.cos(lat1) * tf.cos(lat2) * tf.sin(lon / 2) * tf.sin(lon / 2)
    d = 2 * tf_atan2(tf.sqrt(a), tf.sqrt(1 - a))
    return REarth * d

#B.11
def haversine_distance_time_difference(w_space = 1, w_time = 1):
    '''
    https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
    '''
    def loss(y_true, y_pred):
        err_space = tf_haversine(y_true[:, 0:2], y_pred[:, 0:2])*w_space
        err_time  = mean_squared_error(tf.reshape(y_true[:, 2], (-1, 1)), tf.reshape(y_pred[:, 2], (-1, 1)))*w_time
        return K.mean(tf.transpose(tf.reshape(K.concatenate((err_space, err_time)), (2, -1))), axis=-1)
    return loss

#B.12
def dense_batchnorm_activation(model, l, n):
    for _ in range(l):
        model.add(tf.keras.layers.Dense(n))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('elu'))

#B.13
def spaceNNtime(output_shape, norm = None, dropout_prop = 0.25, l = 10, n = 256, loss_function = "edl", w_time = 1, w_space = 1):
    model = tf.keras.Sequential()                                             # Start a fully connected neural network
    model.add(norm)                                                           # Add a normalization layer
    dense_batchnorm_activation(model, l = int(np.floor(l/2)), n = n)          # Add half of the desired layers
    model.add(tf.keras.layers.Dropout(dropout_prop))                          # Add a drop out layer
    dense_batchnorm_activation(model, l = int(np.ceil(l/2)), n = n)           # Add the rest of the desired layers
    [model.add(tf.keras.layers.Dense(output_shape)) for _ in range(2)]        # Add two extra layers for the output

    if loss_function == "edl":                                                # Compile the model deciding on the loss function and the optimizer
        model.compile(optimizer="Adam", loss=euclidean_distance_loss)
    elif loss_function == "hdtd":
        model.compile(optimizer="Adam", loss=haversine_distance_time_difference(w_time = w_time, w_space = w_space))
    elif loss_function == "hd":
        model.compile(optimizer="Adam", loss=tf_haversine)
    elif loss_function == "mse":
        model.compile(optimizer="Adam", loss=mean_squared_error)
    return model


#B.6
def train_spaceNNtime(model, tra_fea, tra_lab, val_fea, val_lab, callbacks, tra_sample_weight):#, val_sample_weight):
    history = model.fit(x                    = tra_fea, 
                        y                    = tra_lab,
                        epochs               = 5000,
                        batch_size           =   32, 
                        shuffle              = True,
                        verbose              = False,
                        validation_data      = (val_fea, val_lab),#, val_sample_weight),
                        callbacks            = callbacks,
                        sample_weight        = tra_sample_weight)
    
    return history

def train_spaceNNtime_datagen(model, tra_gen, val_gen, callbacks):
    history = model.fit(x                    = tra_gen, 
                        epochs               = 5000,
                        verbose              = False,
                        validation_data      = val_gen,
                        callbacks            = callbacks)
    
    return history

#B.7
def plot_loss(history, fig_path = None):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    if fig_path:
        fig  = plt.figure()
    plt.axvline(x = int(hist[np.min(hist["val_loss"]) == hist["val_loss"]].head(1)["epoch"]), c = "red", alpha = 0.5, linestyle = "dashed", label='Early Stop')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if fig_path:
        fig.savefig(fig_path)
        plt.close()

#B.9
def write_pred(sim, exp, nam, typ, gro, ind, idx, snp, run, pre, true, pred, new_file, file_name):

    df1 = pd.DataFrame({
       "sim" : sim,
       "exp" : exp,
       "nam" : nam,
       "typ" : typ,
       "gro" : gro,
       "ind" : ind,
       "idx" : idx,
       "snp" : snp,
       "run" : run,
       })

    df2 = {}
    if pre in ["sNNt", "space"]:
        df2["true_lat"]   = true[:, 0]
        df2["true_lon"]   = true[:, 1]
        df2["pred_lat"]   = pred[:, 0]
        df2["pred_lon"]   = pred[:, 1]
        df2["diff_space"] = np.array(tf_haversine(true[:, 0:2].astype('float32'), pred[:, 0:2].astype('float32')))

    if pre in ["sNNt", "time"]:
        df2["true_tim"]  = true[:, -1]
        df2["pred_tim"]  = pred[:, -1]
        df2["diff_time"] = true[:, -1]-pred[:, -1]
    df2 = pd.DataFrame(df2)

    df = pd.concat([df1, df2], axis=1)

    if new_file:
        df.to_csv(file_name, mode='w', header=True, sep = "\t", index = False)
    else:
        df.to_csv(file_name, mode='a', header=False, sep = "\t", index = False)
    return False

#B.9
def write_pred_AADR(sim, exp, nam, typ, cro, sta, end, gro, ind, idx, snp, run, pre, true, pred, new_file, file_name):

    df1 = pd.DataFrame({
       "sim" : sim,
       "exp" : exp,
       "nam" : nam,
       "typ" : typ,
       "cro" : cro,
       "sta" : sta,
       "end" : end,
       "gro" : gro,
       "ind" : ind,
       "idx" : idx,
       "snp" : snp,
       "run" : run,
       })

    df2 = {}
    if pre in ["sNNt", "space"]:
        df2["true_lat"]   = true[:, 0]
        df2["true_lon"]   = true[:, 1]
        df2["pred_lat"]   = pred[:, 0]
        df2["pred_lon"]   = pred[:, 1]
        df2["diff_space"] = np.array(tf_haversine(true[:, 0:2].astype('float32'), pred[:, 0:2].astype('float32')))

    if pre in ["sNNt", "time"]:
        df2["true_tim"]  = true[:, -1]
        df2["pred_tim"]  = pred[:, -1]
        df2["diff_time"] = true[:, -1]-pred[:, -1]
    df2 = pd.DataFrame(df2)

    df = pd.concat([df1, df2], axis=1)

    if new_file:
        df.to_csv(file_name, mode='w', header=True, sep = "\t", index = False)
    else:
        df.to_csv(file_name, mode='a', header=False, sep = "\t", index = False)
    return False

#B.11
def write_qc_ind(ind, geno, file_name):
    pd.DataFrame({
        "ind" : ind,
        "noncal" : np.sum(geno == -1, axis = 1),
        "homref" : np.sum(geno ==  0, axis = 1),
        "hethet" : np.sum(geno ==  1, axis = 1),
        "homalt" : np.sum(geno ==  2, axis = 1),
        "noncha" : np.sum((geno > 2) * (geno < -1), axis = 1)
    }).to_csv(file_name, mode='w', header=False, sep = "\t", index = False)

#B.11
def write_qc_snp(snp, geno, file_name):
    pd.DataFrame({
        "snp"    : snp,
        "noncal" : np.sum(geno == -1, axis = 0),
        "homref" : np.sum(geno ==  0, axis = 0),
        "hethet" : np.sum(geno ==  1, axis = 0),
        "homalt" : np.sum(geno ==  2, axis = 0),
        "noncha" : np.sum((geno > 2) * (geno < -1), axis = 0)
    }).to_csv(file_name, mode='w', header=False, sep = "\t", index = False)
