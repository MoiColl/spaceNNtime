import sys
import pyslim
import pandas as pd
sys.path.append('scripts/')
from spaceNNtime_templates import *


sim, exp, nam, met, snp, pre, typ, cov, err, nod = sys.argv[1:]
snp = float(snp)
cov = float(cov)
err = float(err)
nod = int(nod)

ts            = pyslim.load("/home/moicoll/spaceNNtime/data/{sim}/tree.trees".format(sim = sim))
metadata      = pd.read_csv("/home/moicoll/spaceNNtime/data/{sim}/metadata/{met}.txt".format(sim = sim, met = met), delimiter = "\t")
input         = get_input(ts, metadata, snp, typ, cov, err)

output        = get_output(pre, metadata)
print("getting travaltes")
tra_val_tes   = get_tra_val_tes(metadata["ind_id"].to_numpy(), file = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/tra_val_tes_{exp}.json".format(sim = sim, exp = exp))
print("Done")

new_file = True
for i in range(len(tra_val_tes)):
    i = str(i)
    print("Processing batch {}".format(i), flush = True)
    #Normalization
    #if norm == "True":
    mean, std  = mean_sd_Znorm(input[:, tra_val_tes[i]["tra"]])
    input_norm = Znorm(input, mean, std)
    means = []
    stds  = []
    output_norm = np.empty(output.shape)
    for d in range(output.shape[1]):
        mean, std = mean_sd_Znorm(output[tra_val_tes[i]["tra"], d])
        means.append(mean)
        stds.append(std)
        output_norm[:, d] = Znorm(output[:, d], mean, std)
    # else:
    #     means = []
    #     stds  = []
    #     for d in range(output.shape[1]):
    #         means.append(0)
    #         stds.append(1)


    tf.random.set_seed(1234)
    #spaceNNtime
    model             = load_network(input_shape = input.shape[0], output_shape = output.shape[1])
    weights_file_name = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/models/group{i}_weights.hdf5".format(sim = sim, exp = exp, i = i)


    checkpointer, earlystop, reducelr = load_callbacks(weights_file_name)

    history, model = train_network(model, input_norm[:, tra_val_tes[i]["tra"]].T,
                                          input_norm[:, tra_val_tes[i]["val"]].T,
                                          output_norm[tra_val_tes[i]["tra"]],
                                          output_norm[tra_val_tes[i]["val"]], weights_file_name, checkpointer, earlystop, reducelr)

    plot_file_name = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/history_plots/group{i}_history.png".format(sim = sim, exp = exp, i = i)
    plot_history(history, plot_file_name)

    predict = pred(model, input[:, tra_val_tes[i]["tes"]].T) 

    #Saving predictions
    samples   = metadata["ind_id"].to_numpy()
    file_name = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/pred.txt".format(sim = sim, exp = exp)

    new_file  = write_pred(samples, i, tra_val_tes[i]["tes"], input.shape[0], output_norm[tra_val_tes[i]["tes"]], predict, means, stds, new_file, file_name, sim, exp, nam, pre, typ)
