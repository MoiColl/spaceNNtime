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

#B. Functions
#B.1
def generate_tra_val_tes(samples, n_tes = 3, p_tra = 0.9, max_tes_groups = None):
    '''
    Def:
        This function splits samples into training, validating and test groups. This grouping is repeated such that all samples will be
        part of the test set only once. The number of test samples are going to be equal to "n_tes" variable. Since the total number of samples might not be multiple of
        "n_tes" there might be a few test groups with less samples than "n_tes". Once the test samples are chosen randomly in each iteration, the function splits the 
        rest of the samples into train and validation set with a proportion "p_tra" and 1-"p_tra".

        For those groups that have less than "n_tes" as test samples, there will be "n_tes" - len(test samples) samples that will be disregarded and will
        not be part of the training nor validation set. This way, we ensure that each sample has been predicted
        with a model that has been trained with the same number of train and validation samples.

        Finally, each supergroup of train, validation and test sets are going to be included in a dictionary which will be outputed in a file.
        If the file is already created, it will be read instead of created again.
    Input: 
        - samples : numpy array (n_samples, ) with the samples IDs to be divided into train, validation and test sets. All samples are going to be included
                    only once in one test set.
        - n_tes   : integer with the test set size. There will be round_up(len(samples)/n_tes) test sets in total.
        - p_tra   : float number [0-1] proportion of training samples to validation samples. The number of training samples is going to be round down 
                    to favor the number of validation samples.
    
    Output: 
        - tra_val_tes : two level dictionary with:
            - i : integer corresponding to the index of a set
            - tra|val|tes : numpy array (n, ) with the training, validation and test set IDs respectively.
    '''

    n_samples   = samples.shape[0]
    n_fake      = n_tes - (n_samples%n_tes)
    n_val       = int(((1-p_tra)*n_samples)+1)
    tra_val_tes = {}

    idx_samples_fake_shuff = np.arange(n_samples+n_fake)
    np.random.shuffle(idx_samples_fake_shuff) 
    idx_samples_fake_shuff = idx_samples_fake_shuff.reshape(-1, n_tes)

    if not max_tes_groups or max_tes_groups > idx_samples_fake_shuff.shape[0]:
        max_tes_groups = idx_samples_fake_shuff.shape[0]
    
    for i in range(max_tes_groups):
        tes                 = idx_samples_fake_shuff[i, :]
        tra_val_tes[str(i)] = {}
        tes                 = tes[tes < n_samples]
        non_tra_val         = tes

        while non_tra_val.shape[0] < n_tes:
            exc = np.random.randint(n_samples, size=1)
            if exc not in non_tra_val:
                non_tra_val = np.append(non_tra_val, exc)

        p = np.full((n_samples, ), 1/(n_samples-n_tes))
        p[non_tra_val] = 0

        val = np.random.choice(np.arange(n_samples), size=n_val, replace=False, p=p)

        tra_bool = np.full((n_samples, ), True)
        tra_bool[np.concatenate([non_tra_val, val])] = False
        tra     = np.arange(n_samples)[tra_bool]
        
        tra.sort()
        val.sort()
        tes.sort()

        tra_val_tes[str(i)]["tra"] = tra.tolist()
        tra_val_tes[str(i)]["val"] = val.tolist()
        tra_val_tes[str(i)]["tes"] = tes.tolist()

        
    return tra_val_tes

#B.2
def get_tra_val_tes(samples, file = "", overwrite = False, n_tes = 3, p_tra = 0.9, max_tes_groups = None):
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
                #tra_val_tes = yaml.load(tra_val_tes_file, Loader=yaml.FullLoader)
        else:
            tra_val_tes = generate_tra_val_tes(samples, n_tes, p_tra, max_tes_groups)
            with open(file, 'w') as tra_val_tes_file:
                json.dump(tra_val_tes, tra_val_tes_file)
                #yaml.dump(tra_val_tes, tra_val_tes_file, default_flow_style=False)
    else:
        tra_val_tes = generate_tra_val_tes(samples, n_tes, p_tra, max_tes_groups)
        
    return tra_val_tes

#B.3
def get_input(ts, metadata, snp, typ, cov, err):
    '''
    Def :
        Obtain the allele counts for variant sites from a simulated tree structure (ts) that are not singletons or nearly fixed for a subset 
        of individuals indicated in a metadata pandas DataFrame.
    Input :
        - ts       : tree data structure
        - metadata : pandas DataFrame (n_individuals, features) in which for each individual there are indicated both nodes ID in the ts in
                     columns named "node1" and "node2"
    Output :
        -  allele_counts : numpy array (n_variants, n_individuals) with allele counts per individual.
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
        return gm
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

        return GLmix.T
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

        return GPmix.T
    elif typ == "gl":
        arc      = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL       = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        return GL.transpose((1, 0, 2)).reshape(-1).reshape(GL.shape[1], GL.shape[0]*3).T
    elif typ == "gp":
        arc        = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL         = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        likelihood = np.exp(-GL)
        prior      = np.array([1/4, 1/2, 1/4])
        GP         = likelihood*prior/(np.sum(likelihood*prior, axis = 2).reshape(GL.shape[:2] + (1,)))
        return GP.transpose((1, 0, 2)).reshape(-1).reshape(GP.shape[1], GP.shape[0]*3).T
    elif typ == "gl_gt":
        arc      = simGL.sim_allelereadcounts(gm = gm, mean_depth = cov, std_depth = 1, e = err, ploidy = 2, seed = 1234)
        GL       = simGL.allelereadcounts_to_GL(arc = arc, e = err, ploidy = 2)[:, :, [0, 1, 4]]
        
        minGLidx = np.argsort(GL)[:, :, 0]
        
        missing  = ((GL == 0).sum(axis = 2) == 3)
        minGLidx[missing] = -1

        return minGLidx

def get_input_AADR(metadata, chrom, start, end):
    snp     = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.snp", index_col = None, header = None, names = ["snp", "chr", "gen", "pos", "ref", "alt"])

    snp_idx = np.array([i for i in snp[(snp.chr == chrom) & (snp.pos >= start) & (snp.pos < end)].index])
    geno = []
    with open("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public.eigenstratgeno", "r") as f:
        for i, l in enumerate(f):
            if i >= snp_idx[0] and i <= snp_idx[-1]:
                geno.append([int(x) for x in l.strip()])
            elif i > snp_idx[-1]:
                break
            
    geno = np.array(geno)[:, metadata.index.to_numpy()]

    geno[geno == 9] = -1

    return geno, snp[(snp.chr == chrom) & (snp.pos >= start) & (snp.pos < end)].pos.to_numpy()


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

def normalizer(nor, input_shape):
    if nor == "None":
        return tf.keras.layers.Normalization(input_shape=[input_shape, ], axis = None, mean = 0, variance = 1)
    elif nor == "Norm0":
        return tf.keras.layers.Normalization(input_shape=[input_shape, ], axis = None)
    elif nor == "Norm1":
        return tf.keras.layers.Normalization(input_shape=[input_shape, ], axis = -1)

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
def train_spaceNNtime(model, tra_fea, tra_lab, val_fea, val_lab, callbacks, sample_weight):
    history = model.fit(x                    = tra_fea, 
                        y                    = tra_lab,
                        epochs               = 5000,
                        batch_size           =   32,
                        shuffle              = True,
                        verbose              = False,
                        validation_data      = (val_fea, val_lab),
                        callbacks            = callbacks,
                        sample_weight        = sample_weight)
    
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
        "noncal" : np.sum(geno == -1, axis = 0),
        "homref" : np.sum(geno ==  0, axis = 0),
        "hethet" : np.sum(geno ==  1, axis = 0),
        "homalt" : np.sum(geno ==  2, axis = 0),
        "noncha" : np.sum((geno > 2) * (geno < -1), axis = 0)
    }).to_csv(file_name, mode='w', header=False, sep = "\t", index = False)

#B.11
def write_qc_snp(snp, geno, file_name):
    print(snp.shape, np.sum(geno == -1, axis = 1).shape)
    pd.DataFrame({
        "snp"    : snp,
        "noncal" : np.sum(geno == -1, axis = 1),
        "homref" : np.sum(geno ==  0, axis = 1),
        "hethet" : np.sum(geno ==  1, axis = 1),
        "homalt" : np.sum(geno ==  2, axis = 1),
        "noncha" : np.sum((geno > 2) * (geno < -1), axis = 1)
    }).to_csv(file_name, mode='w', header=False, sep = "\t", index = False)