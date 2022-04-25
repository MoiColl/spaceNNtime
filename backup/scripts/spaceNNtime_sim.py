import sys
import pyslim
import pandas as pd
sys.path.append('scripts/')
from spacetime_templates import *

exp, data_type, norm, ys, p = sys.argv[1:]
p = float(p)


ts            = pyslim.load("sandbox/europe/processed_tree/tree.trees")
metadata      = pd.read_csv("sandbox/europe/spaceNNtime/exp{exp}/metadata.csv".format(exp = exp))
allele_counts = allele_counts_from_simulations(ts, metadata, p)
tra_val_tes   = get_tra_val_tes(None, file = "sandbox/europe/spaceNNtime/exp{exp}/exp{exp}.yaml".format(exp = exp))

# print(ts)
# print("metadata.shape")
# print(metadata.shape)
# print("metadata")
# print(metadata)
# print("allele_counts")
# print(allele_counts)
# print("len(tra_val_tes)")
# print(len(tra_val_tes))
# print("len(tra_val_tes[0])")
# print(len(tra_val_tes[0]))
# print("len(tra_val_tes[0]['tra'])")
# print(len(tra_val_tes[0]['tra']))
# print(len(tra_val_tes[0]['val']))
# print(len(tra_val_tes[0]['tes']))
# sys.exit()

if ys == "spaceNNtime":
    lat_long_time = metadata[["loc1", "loc2", "time"]].to_numpy()
elif ys == "space":
    lat_long_time = metadata[["loc1", "loc2"]].to_numpy()
elif ys == "time":
    lat_long_time = metadata[["time"]].to_numpy()
else:
    sys.exit("Ys value incorrect!\n")

new_file = True

for i in range(len(tra_val_tes)):
    print("Processing batch {}".format(i), flush = True)
    
    #Normalization
    if norm == "True":
        mean, std         = mean_sd_Znorm(allele_counts[:, tra_val_tes[i]["tra"]])
        allele_counts_tmp = Znorm(allele_counts, mean, std)

        means = []
        stds  = []
        lat_long_time_tmp = np.empty(lat_long_time.shape)
        for d in range(lat_long_time.shape[1]):
            mean, std = mean_sd_Znorm(lat_long_time[tra_val_tes[i]["tra"], d])
            means.append(mean)
            stds.append(std)
            lat_long_time_tmp[:, d] = Znorm(lat_long_time[:, d], mean, std)
    else:
        means = []
        stds  = []
        for d in range(lat_long_time.shape[1]):
            means.append(0)
            stds.append(1)


    tf.random.set_seed(1234)
    #spaceNNtime
    model             = load_network(input_shape = allele_counts.shape[0], output_shape = lat_long_time.shape[1])
    weights_file_name = "sandbox/europe/spaceNNtime/exp{exp}/models/group{i}_weights.hdf5".format(exp = exp, i = i)

    checkpointer, earlystop, reducelr = load_callbacks(weights_file_name)

    history, model = train_network(model, allele_counts_tmp[:, tra_val_tes[i]["tra"]].T,
                                          allele_counts_tmp[:, tra_val_tes[i]["val"]].T,
                                          lat_long_time_tmp[tra_val_tes[i]["tra"]],
                                          lat_long_time_tmp[tra_val_tes[i]["val"]], weights_file_name, checkpointer, earlystop, reducelr)

    plot_file_name = "sandbox/europe/spaceNNtime/exp{exp}/history_plots/group{i}_history.png".format(exp = exp, i = i)
    plot_history(history, plot_file_name)

    predict = pred(model, allele_counts[:, tra_val_tes[i]["tes"]].T) 


    #Saving predictions
    samples   = metadata["inid"].to_numpy()
    file_name = "sandbox/europe/spaceNNtime/exp{exp}/pred.txt".format(exp = exp)

    new_file  = write_pred(samples, "ac", ys, i, tra_val_tes, allele_counts.shape[0], lat_long_time_tmp[tra_val_tes[i]["tes"]], predict, means, stds, new_file, file_name)











