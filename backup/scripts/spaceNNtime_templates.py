#A. Import
import numpy      as np
import pandas     as pd
import os
import sys
import json
#import yaml
from matplotlib  import pyplot as plt
import tensorflow as tf
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

#B.4
def mean_sd_Znorm(a):
    return a.mean(), a.std()

#B.5
def Znorm(a, mean, std):
    return (a-mean)/std

#B.6
def load_callbacks(weights_file_name):
    checkpointer=tf.keras.callbacks.ModelCheckpoint(
                      filepath=weights_file_name,
                      verbose=False,
                      save_best_only=True,
                      save_weights_only=True,
                      monitor="val_loss",
                      save_freq="epoch")

    earlystop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                               min_delta=0,
                                               patience=100)
    
    reducelr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.5,
                                                  patience=int(100/6),
                                                  verbose=False,
                                                  mode='auto',
                                                  min_delta=0,
                                                  cooldown=0,
                                                  min_lr=0)
    return checkpointer,earlystop,reducelr


#B.5
def load_network(input_shape, output_shape, dropout_prop = 0.25, nlayers = 10, width = 256):
    from tensorflow.keras import backend as K
    def euclidean_distance_loss(y_true, y_pred):                              #Computes euclidian distance between coordinates
        return K.sqrt(K.sum(K.square(y_pred - y_true),axis=-1))
    
    model = tf.keras.Sequential()                                             #Start a fully connected neural network
    
    
    #model.add(tf.keras.layers.BatchNormalization(input_shape=(input_shape,))) #Normalization of the input
    
    
    for i in range(int(np.floor(nlayers/2))):
        model.add(tf.keras.layers.Dense(width, activation="elu"))             #Half layers fully connected (elu)
        
    model.add(tf.keras.layers.Dropout(dropout_prop))                          #Perform dropout
    
    for i in range(int(np.ceil(nlayers/2))):
        model.add(tf.keras.layers.Dense(width, activation="elu"))             #The other half of fully connected layers (elu)
    
    model.add(tf.keras.layers.Dense(output_shape))                                       #One layer with two nodes
    
    model.add(tf.keras.layers.Dense(output_shape))                                       #Another layer with two nodes
    
    model.compile(optimizer="Adam",
                  loss=euclidean_distance_loss)
    return model


#B.6
def train_network(model, tra_aco, val_aco, tra_loc, val_loc, weights_file_name, checkpointer, earlystop, reducelr):
    history = model.fit(tra_aco, tra_loc,
                        epochs=5000,
                        batch_size=32,
                        shuffle=True,
                        verbose=False,
                        validation_data=(val_aco, val_loc),
                        callbacks=[checkpointer,earlystop,reducelr])
    
    model.load_weights(weights_file_name)
    
    return history, model



#B.7
def plot_history(history, fig_path):
    
    fig= plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    fig.savefig(fig_path)
    plt.close()


#B.8
def pred(model, tes_aco):
    return model.predict_step(tes_aco)
     #model.predict(tes_aco)  

#B.9
def write_pred(samples, i, tes, n_snps, tes_loc, predict, means, stds, new_file, file_name, sim, exp, nam, pre, typ, run):
    df1 = pd.DataFrame({
       "sim"                 : sim,
       "exp"                 : exp,
       "nam"                 : nam, 
       "ind"                 : samples[tes],
       "typ"                 : typ,
       "group"               : i,
       "index"               : tes,
       "n_snps"              : n_snps,
       "run_time"            : run,
       })

    if pre == "sNNt":
        df2 = pd.DataFrame({
            "real_latitude_norm"  : tes_loc[:, 0],
            "real_longitude_norm" : tes_loc[:, 1],
            "real_time_norm"      : tes_loc[:, 2],
            "pred_latitude_norm"  : predict[:, 0],
            "pred_longitude_norm" : predict[:, 1],
            "pred_time_norm"      : predict[:, 2],
            "real_latitude"       : (tes_loc[:, 0]*stds[0])+means[0],
            "real_longitude"      : (tes_loc[:, 1]*stds[1])+means[1],
            "real_time"           : (tes_loc[:, 2]*stds[2])+means[2],
            "pred_latitude"       : (predict[:, 0]*stds[0])+means[0],
            "pred_longitude"      : (predict[:, 1]*stds[1])+means[1],
            "pred_time"           : (predict[:, 2]*stds[2])+means[2]
       })
    
    elif pre == "space":
        df2 = pd.DataFrame({
            "real_latitude_norm"  : tes_loc[:, 0],
            "real_longitude_norm" : tes_loc[:, 1],
            "pred_latitude_norm"  : predict[:, 0],
            "pred_longitude_norm" : predict[:, 1],
            "real_latitude"       : (tes_loc[:, 0]*stds[0])+means[0],
            "real_longitude"      : (tes_loc[:, 1]*stds[1])+means[1],
            "pred_latitude"       : (predict[:, 0]*stds[0])+means[0],
            "pred_longitude"      : (predict[:, 1]*stds[1])+means[1]
        })

    elif pre == "time":
        df2 = pd.DataFrame({
            "real_time_norm"  : tes_loc[:, 0],
            "pred_time_norm"  : predict[:, 0],
            "real_time"       : (tes_loc[:, 0]*stds[0])+means[0],
            "pred_time"       : (predict[:, 0]*stds[0])+means[0]
        })

    else:
        sys.exit("pre variable not correct!. It can only be sNNt, time or space and it is {}\n".format(pre))

    df = pd.concat([df1, df2], axis=1)

    if new_file:
        df.to_csv(file_name, mode='w', header=True, sep = "\t", index = False)
    else:
        df.to_csv(file_name, mode='a', header=False, sep = "\t", index = False)
    return False

##B.3
#def deal_with_missing_vec(f_arr, n_missing_arr):
#    '''
#    Def:
#        For a series of sites, it returns all the imputed genotypes as a single 1d array
#    Input:
#        - f_arr         : An array for every SNP with missing genotypes with the fraction of allele 1
#        - n_missing_arr : An array for every SNP with missing genotypes with the number of inidivudals with missing genotypes
#    Output:
#        - numpy array (1d) with all the missing genotypes
#    '''
#    genotypes = []
#    for f, n_missing in zip(f_arr, n_missing_arr):
#        genotypes += deal_with_missing(f, n_missing)
#    return np.array(genotypes, dtype="int8")
#
##B.4
#def impute_missing_data(genotypes, missing, count_alleles):
#    '''
#    Input:
#        - genotypes : genotype array with missing genotypes
#    Output:
#        - genotypes : genotype array with no missing genotypes
#    '''
#    genotypes[missing] = deal_with_missing_vec(count_alleles[:, 0]/np.sum(count_alleles, axis = 1), np.sum(missing, axis = 1))
#    
#    return genotypes
#
#
##B.5
#def deal_with_missing(f, n_missing):
#    '''
#    Def:
#        For a given SNP with a certain number individuals with missing genotype and an allele fraction f, it returns
#        imputed genotypes for those individuals.
#    Input: 
#        - f         : Fraction of allele 0
#        - n_missing : number of individuals with missing genotype
#    Output:
#        - list with inputed missing alleles
#
#    '''
#    c = {0 : [0, 0],
#         1 : [0, 1],
#         2 : [1, 1]}
#    return [c[x] for x in np.random.binomial(2, f, n_missing)]
#
##B.6
#def get_GLdict(n_alt):
#    '''
#    Def:
#        It first create a dictionary (c) with the genotype index with the value equal to the sorted position that correspond in the 
#        GL matrix (00 : 0, 01 : 1, 11 : 2, 02 : 3, 12, 22...). Then, knowing this info, it creates a two level dictionary, each level
#        corresponding to the allele index, that returns a list with the indeces in the GL matrix that must be taken to get the correct
#        GL given two allele indeces. 
#    Input:
#        - n_alt : number of alternative alleles
#    Output: 
#        - z     :  
#    '''
#    c = defaultdict(lambda : None)
#    z = defaultdict(lambda : defaultdict(lambda : []))
#    i = 0
#    for x in range(n_alt):
#        for y in range(x+1):
#            c[str(y)+str(x)] = i
#            i+=1
#
#    for x in range(n_alt):
#        for y in range(0, x):
#            z[y][x].append(c[str(y)+str(y)])
#            z[y][x].append(c[str(y)+str(x)])
#            z[y][x].append(c[str(x)+str(x)])
#
#    return z
#
##B.7 
#def get_filters_and_index(callset, chrom, start, end):
#    '''
#    Def:
#        It checks which SNPs (given a chromosome and a start and end index) are biallelic and singletons and computes 
#        the count_alleles and the allele_index for those SNPs that are biallelic and not singletons.
#    Input:
#        - callset : zarr callset variable
#        - chrom   : chromosome 
#        - start   : start index (not genomic position)
#        - end     : ending index (not genomic position)
#    Output:
#        - filterp       : boolean np array (n_SNPs, ) with information if a SNP passes the SNP filter or not
#        - biallel       : boolean np array (filterp n_SNPs, ) with information if a SNP is biallelic or not
#        - singlet       : boolean np array (biallelic SNPs, ) with information if a biallelic SNP is not singleton (True) or it is (False)
#        - count_alleles : np array (n_SNPs, n_alleles) with the counts of alleles for each SNP counting all individuals
#        - allele_index  : np array (n_SNPs, 2) with the index of the allele index that it's counts are > 0 (because we are looking at biallelic positions, we 
#                          only have to look at 2 positions)
#    '''
#
#    filterp       = callset['{}/variants/FILTER_PASS'.format(chrom)][start:end]
#    
#    biallel       = (allel.GenotypeDaskArray(callset['{}/calldata/GT'.format(chrom)])
#                         .take(np.arange(start, end), axis = 0)
#                         .compress(filterp, axis=0)
#                         .count_alleles()
#                         .is_biallelic()
#                         .compute())
#    
#    singlet       = ~(allel.GenotypeDaskArray(callset['{}/calldata/GT'.format(chrom)])
#                         .take(np.arange(start, end), axis = 0)
#                         .compress(filterp, axis=0)
#                         .compress(biallel, axis=0)
#                         .count_alleles()
#                         .is_singleton()
#                         .compute())
#
#    count_alleles = (allel.GenotypeDaskArray(callset['{}/calldata/GT'.format(chrom)])
#                         .take(np.arange(start, end), axis = 0)
#                         .compress(filterp, axis=0)
#                         .compress(biallel, axis=0)
#                         .compress(singlet, axis=0)
#                         .count_alleles()
#                         .compute())
#        
#    allele_index  = np.where(count_alleles != 0)[1].reshape(-1, 2) # array with the index of the alleles segregating in the population
#
#    return filterp, biallel, singlet, count_alleles, allele_index
#
#
#
#
##B.8
#def create_nninput(callset, chrom, start_pos_idx, end_pos_idx, pos, input_type, noncall = -1):
#    '''
#    Def:
#        This function will output the genotype or genotype likelihoods and position array for those SNPs that are biallelic and are not singletons.
#        If genotypes are requested, the missing genotypes are going to be inputed and only the allele counts with the minimum allele index (ref : 0,
#        alt1 : 1, alt2 : 2...) will be kept since keeping the allele counts for both alleles is redundant.
#        If genotype likelihoods are requested, 3 genotype likelihoods (homo x 2, hetero) are going to be outputed for each SNP. In this case, no
#        imputing is performed.
#
#    Input:
#        - callset       : scikit allel callset valriable
#        - chrom         : chromosome
#        - start_pos_idx : starting position index
#        - end_pos_idx   : ending position index
#        - pos           : position SortedIndex() allel list of the window to be examined
#        - input_type    : "ac" : alternative allele counts ;"gl" : 3 genotype likelyhoods; or "gl_mix": allele count and the second most likely genotype likelyhood.        - noncall       : How the missing data is encoded. Either 0 or -1.
#    Output: 
#        - nninput               : np array with dimentions (n_SNPs, n_ind) if genotypes, (n_SNPs x 3, n_ind) if genotype likelihoods
#        - pos[biallel][singlet] : SNP genomic possitions that are biallelic and not singletons 
#    '''
#    filterp, biallel, singlet, count_alleles, allele_index = get_filters_and_index(callset, chrom, start_pos_idx, end_pos_idx)
#    
#    mapping = np.full(count_alleles.shape, -1)
#    mapping[np.arange(count_alleles.shape[0]), allele_index[:, 0]] = np.full(count_alleles.shape[0], 0)
#    mapping[np.arange(count_alleles.shape[0]), allele_index[:, 1]] = np.full(count_alleles.shape[0], 1)
#
#    genotypes = (allel.GenotypeDaskArray(callset['{}/calldata/GT'.format(chrom)])
#                             .take(np.arange(start_pos_idx, end_pos_idx), axis = 0)
#                             .compress(filterp, axis=0)
#                             .compress(biallel, axis=0)
#                             .compress(singlet, axis=0)
#                             .map_alleles(mapping)
#                             .compute())
#    
#    missing   = genotypes.is_missing()
#
#    
#    if input_type == "gl" or input_type == "gl_mix":
#
#        GLd      = get_GLdict(8)
#        genolike = None
#        w        = int(np.ceil((end_pos_idx-start_pos_idx)/4))
#
#        for s in range(start_pos_idx, end_pos_idx, w):
#            if s+w < end_pos_idx:
#                e = s+w
#            else:
#                e = end_pos_idx
#
#            filterp_sub, biallel_sub, singlet_sub, _, allele_index_sub = get_filters_and_index(callset, chrom, s, e)
#
#            GL_index = np.array([GLd[allele_index_sub[i][0]][allele_index_sub[i][1]] for i in range(allele_index_sub.shape[0])])
#
#            n_pos    = GL_index.shape[0]
#            n_ind    = callset['{}/samples'.format(chrom)].shape[0]
#            n_gl     = GL_index.shape[1]
#            
#            
#            
#            genolike_sub = (callset['{}/calldata/PL'.format(chrom)][s:e, :, :][filterp_sub][biallel_sub][singlet_sub][np.repeat(np.arange(n_pos), n_ind*n_gl), np.repeat(np.tile(np.arange(n_ind), n_pos), n_gl), np.tile(GL_index, n_ind).reshape(-1)]
#                                    .reshape(n_pos, n_ind, n_gl))
#            
#            if noncall == 0:
#                genolike_sub[genolike_sub == -1] = 0 
#
#            if type(genolike) == type(None):
#                genolike = genolike_sub
#            else:
#                genolike = np.concatenate((genolike, genolike_sub), axis = 0)
#                
#        genolike[missing] = noncall
#                
#        if input_type == "gl_mix":
#            ac           = np.full((genotypes.shape[0], genotypes.shape[1]), noncall)   
#            ac[~missing] = genotypes.to_allele_counts()[:, :, 0][~missing]
#            ac           = ac.reshape(genotypes.shape[0], genotypes.shape[1], 1)
#            
#            median       = np.median(genolike, axis = 2).reshape(genolike.shape[0], genolike.shape[1], 1)
#            
#            genolike_mix = np.concatenate((median, ac), axis = 2)
#    
#    if   input_type == "ac":
#        return impute_missing_data(genotypes, missing, count_alleles).to_allele_counts()[:, :, 0], pos[filterp][biallel][singlet]
#    elif input_type == "gl":
#        return genolike.transpose(0, 2, 1).reshape(-1, n_ind),                      pos[filterp][biallel][singlet]
#    elif input_type == "gl_mix":
#        return genolike_mix.transpose(0, 2, 1).reshape(-1, n_ind),                  pos[filterp][biallel][singlet]
#    else:
#        sys.exit("No such inputtype: {}".format(input_type))
#
#
##B.9
#def load_callbacks(weights_file_name):
#    checkpointer=tf.keras.callbacks.ModelCheckpoint(
#                      filepath=weights_file_name,
#                      verbose=False,
#                      save_best_only=True,
#                      save_weights_only=True,
#                      monitor="val_loss",
#                      save_freq="epoch")
#
#    earlystop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
#                                               min_delta=0,
#                                               patience=100)
#    
#    reducelr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                                  factor=0.5,
#                                                  patience=int(100/6),
#                                                  verbose=False,
#                                                  mode='auto',
#                                                  min_delta=0,
#                                                  cooldown=0,
#                                                  min_lr=0)
#    return checkpointer,earlystop,reducelr
#
#
##B.10
#def load_network(input_shape, dropout_prop = 0.25, nlayers = 10, width = 256):
#    from tensorflow.keras import backend as K
#    def euclidean_distance_loss(y_true, y_pred):                              #Computes euclidian distance between coordinates
#        return K.sqrt(K.sum(K.square(y_pred - y_true),axis=-1))
#    
#    model = tf.keras.Sequential()                                             #Start a fully connected neural network
#    
#    
#    model.add(tf.keras.layers.BatchNormalization(input_shape=(input_shape,))) #Normalization of the input
#    
#    
#    for i in range(int(np.floor(nlayers/2))):
#        model.add(tf.keras.layers.Dense(width, activation="elu"))             #Half layers fully connected (elu)
#        
#    model.add(tf.keras.layers.Dropout(dropout_prop))                          #Perform dropout
#    
#    for i in range(int(np.ceil(nlayers/2))):
#        model.add(tf.keras.layers.Dense(width, activation="elu"))             #The other half of fully connected layers (elu)
#    
#    model.add(tf.keras.layers.Dense(2))                                       #One layer with two nodes
#    
#    model.add(tf.keras.layers.Dense(2))                                       #Another layer with two nodes
#    
#    model.compile(optimizer="Adam",
#                  loss=euclidean_distance_loss)
#    return model
#
##B.11
#def train_network(model, tra_aco, val_aco, tra_loc, val_loc, weights_file_name, checkpointer, earlystop, reducelr):
#    history = model.fit(tra_aco, tra_loc,
#                        epochs=5000,
#                        batch_size=32,
#                        shuffle=True,
#                        verbose=False,
#                        validation_data=(val_aco, val_loc),
#                        callbacks=[checkpointer,earlystop,reducelr])
#    
#    model.load_weights(weights_file_name)
#    
#    return history, model
#
##B.12
#def pred(model, tes_aco):
#    return model.predict_step(tes_aco)
#     #model.predict(tes_aco)  
#
##B.13
#def plot_history(history, fig_path):
#    
#    fig= plt.figure()
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'validation'], loc='upper right')
#    fig.savefig(fig_path)
#    plt.close()
#
##B.14
#def dist_degrees(real_lat, pred_lat, real_lon, pred_lon):
#    '''
#    Def:
#        Returns the distance in Km between two geographical points in latitude and longitude degrees using the havershine formula
#        https://www.movable-type.co.uk/scripts/latlong.html
#    Input:
#        - real_lat : real geographical location of a sample in latitude degrees
#        - pred_lat : predicted geographical location of a sample in latitude degrees
#        - real_lon : real geographical location of a sample in longuitude degrees
#        - pred_lon : predicted geographical location of a sample in longuitude degrees
#    Output:
#        - Distance between the two points in Km
#    '''
#    R        = 6371   #Earth radius in Km
#
#    rad_real_lat = real_lat*(np.pi/180) #real latitude in radiants
#    rad_pred_lat = pred_lat*(np.pi/180) #predicted latitude in radiants
#    diff_lat = (pred_lat-real_lat)*(np.pi/180) #difference between latitudes in radiants
#    diff_lon = (pred_lon-real_lon)*(np.pi/180) #difference between longuitudes in radiants
#    a = np.power(np.sin(diff_lat/2), 2) + (np.cos(rad_real_lat) * np.cos(rad_pred_lat) * np.power(np.sin(diff_lon/2), 2)) #Trigonometry :)
#    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#
#    return R*c
#
#
##B.15
#def write_data(samples, input_type, i, tra_val_tes, chrom,  s, window, n_snps, tes_loc, predict, std_lat, mean_lat, std_lon, mean_lon, new_file, file_name):
#    df = pd.DataFrame({
#        "samples"             : samples[tra_val_tes[i]["tes"]],
#        "input_type"          : input_type,
#        "group"               : i,
#        "index"               : tra_val_tes[i]["tes"],
#        "chrom"               : chrom,
#        "start"               : s,
#        "end"                 : s+window,
#        "n_snps"              : n_snps,
#        "real_latitude_norm"  : tes_loc[:, 0],
#        "real_longitude_norm" : tes_loc[:, 1],
#        "pred_latitude_norm"  : predict[:, 0],
#        "pred_longitude_norm" : predict[:, 1],
#        "real_latitude"       : (tes_loc[:, 0]*std_lat)+mean_lat,
#        "real_longitude"      : (tes_loc[:, 1]*std_lon)+mean_lon,
#        "pred_latitude"       : (predict[:, 0]*std_lat)+mean_lat,
#        "pred_longitude"      : (predict[:, 1]*std_lon)+mean_lon
#    }).assign(dist = lambda x: dist_degrees(x.real_latitude, x.pred_latitude, x.real_longitude, x.pred_longitude))
#    if new_file:
#        df.to_csv(file_name, mode='w', header=True, sep = "\t", index = False)
#    else:
#        df.to_csv(file_name, mode='a', header=False, sep = "\t", index = False)
#    return False
#



















