#A. Import
from collections         import defaultdict
from matplotlib          import pyplot as plt
from spacetime_templates import *
import numpy      as np
import pandas     as pd
import tensorflow as tf
import sys
import gzip
import time
import allel
import zarr
import numcodecs
import time
import os
import yaml
import dask

#B. Code
print("Available devices:\n")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("", flush = True)

start_time   = time.time()
current_time = time.time()

new_file    = True
chrom       = sys.argv[1]     #"1"
start       = int(sys.argv[2])#          0
end         = int(sys.argv[3])#250,000,000
window      = int(sys.argv[4])# 10,000,000
input_type  = sys.argv[5]     #"genoytpe"

print("Processing chromosome {} window {} Mb to {} Mb using {} data - since last : {:.2f} min - since start : {:.2f} min".format(chrom, start/1e6, end/1e6, input_type, (time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
current_time = time.time()

callset     = zarr.open_group("/home/moicoll/GenerationInterval/people/moi/tmp/zarr/HGDP/hgdp_wgs.20190516.full.chr{}.zarr".format(chrom), mode='r')

positions   = callset['{}/variants/POS'.format(chrom)]

samples     = callset['{}/samples'.format(chrom)][:]
tra_val_tes = get_tra_val_tes(samples)

lat_long, mean_lat, mean_lon, std_lat, std_lon = get_lat_long(samples)

for s in range(start, end, window):
    idx_pos = np.where((positions[:] > s) * (positions[:] <= s+window))[0] #idx_pos_prepro

    if len(idx_pos):
        print("  - Processing chromosome {} window {} Mb to {} Mb - since last : {:.2f} min - since start : {:.2f} min".format(chrom, s/1e6, (s+window)/1e6, (time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
        current_time = time.time()
        sta_pos      = idx_pos[0]    #sta_pos_prepro
        end_pos      = idx_pos[-1]+1 #end_pos_prepro
        pos          = positions[sta_pos:end_pos] #pos_prepro
        nninput, pos = create_nninput(callset, chrom, sta_pos, end_pos, pos, input_type)
        for i in range(10):#range(tra_val_tes):
            print("    + Group {} : {:.2f} min - since start : {:.2f} min".format(i, (time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
            current_time = time.time()
            print("      · Creating model with input size {} - since last : {:.2f} min - since start : {:.2f} min".format(nninput.shape[0], (time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
            current_time = time.time()
            tf.keras.backend.clear_session()
            model = load_network(input_shape = nninput.shape[0])
            weights_file_name = "/home/moicoll/GenerationInterval/people/moi/spacetime/tmp/models/HGDP_chr{}_{}MB_{}_{}_{}_group{}weights.hdf5".format(chrom, int(window/1e6), int(start/window), int(end/window), input_type, i)
            checkpointer,earlystop,reducelr = load_callbacks(weights_file_name)
            print("      · Training model - since last : {:.2f} min - since start : {:.2f} min".format((time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
            current_time = time.time()
            history, model = train_network(model, nninput[:, tra_val_tes[i]["tra"]].T,
                                                  nninput[:, tra_val_tes[i]["val"]].T,
                                                  lat_long[tra_val_tes[i]["tra"]],
                                                  lat_long[tra_val_tes[i]["val"]], weights_file_name, checkpointer, earlystop, reducelr)
            plot_file_name = "/home/moicoll/GenerationInterval/people/moi/spacetime/tmp/history_plots/HGDP_chr{}_{}MB_{}_{}_{}_group{}.png".format(chrom, int(window/1e6), int(start/window), int(end/window), input_type, i)
            plot_history(history, plot_file_name)
            
            print("      · Predict - since last : {:.2f} min - since start : {:.2f} min".format((time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
            current_time = time.time()
            predict = pred(model, nninput[:, tra_val_tes[i]["tes"]].T)  
            
            print("      · Writing results - since last : {:.2f} min - since start : {:.2f} min".format((time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
            current_time = time.time()                          
            
            file_name = "/home/moicoll/GenerationInterval/people/moi/spacetime/tmp/spacetime_predictions/HGDP_chr{}_{}MB_{}_{}_{}.tmp".format(chrom, int(window/1e6), int(start/window), int(end/window), input_type)
            new_file = write_data(samples, input_type, i, tra_val_tes, chrom, s, window, idx_pos.shape[0], lat_long[tra_val_tes[i]["tes"]], predict, std_lat, mean_lat, std_lon, mean_lon, new_file, file_name)

            del model
        del nninput
    else:
        print("  - No SNPs for chromosome {} window {} Mb to {} Mb - since last : {:.2f} min - since last : {:.2f} min".format(chrom, s/1e6, (s+window)/1e6, (time.time()-current_time)/60, (time.time()-start_time)/60), flush = True)
        current_time = time.time()
               
print("Done - {:.2f} min - {:.2f} min".format((time.time()-current_time)/60, (time.time()-start_time)/60))




















