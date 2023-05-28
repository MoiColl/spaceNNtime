import sys
import tskit
import pandas as pd
import time
import os
sys.path.append('scripts/')
from spaceNNtime_templates import *
from tensorflow.python.client import device_lib

print("Which devices are available?")
print(device_lib.list_local_devices())

sim, exp, nam, met, snp, pre, lay, dro, typ, cov, std, err, los, nfe, nla, wti, wsp, wsa, nod, dat = sys.argv[1:]
snp = float(snp)
lay = int(lay)
dro = float(dro)
cov = float(cov)
std = float(std)
err = float(err)
nod = int(nod)
wti = float(wti)
wsp = float(wsp)


ts            = tskit.load("/home/moicoll/spaceNNtime/data/{sim}/tree.trees".format(sim = sim))
metadata      = pd.read_csv("/home/moicoll/spaceNNtime/data/{sim}/metadata/{met}.txt".format(sim = sim, met = met), delimiter = "\t")


if nam not in ["loss", "reference", "downsample", "sampling", "snp_density", "prediction", "n_nodes", "dropout", "layers", "fasttest"]:
    cov_file_path = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/coverage.txt".format(sim = sim, exp = exp)
    if os.path.isfile(cov_file_path):
        cov = pd.read_table(cov_file_path)[["co1", "co2"]].to_numpy().reshape(-1)
    else:
        ploidy = 2
        cov = simGL.depth_per_haplotype(rng = np.random.default_rng(1234), mean_depth = cov/ploidy, std_depth = std/ploidy, n_hap = metadata.shape[0]*ploidy, ploidy = ploidy)
        pd.DataFrame({"ind" : metadata["ind_id"],
                    "co1" : cov.reshape(-1, 2)[:, 0],
                    "co2" : cov.reshape(-1, 2)[:, 1]}).to_csv(cov_file_path, mode='w', header=True, sep = "\t", index = False)

if wsa == "None":
    wsa = np.ones(metadata.shape[0])
elif wsa == "coverage":
    wsa = cov.reshape(-1, 2).sum(axis = 1)
elif wsa == "coveragesigmoid":
    wsa = cov.reshape(-1, 2).sum(axis = 1)
    wsa = 1/(1+np.exp((-wsa/4.5)))

input         = get_input_simulated_tree(ts, metadata, snp, typ, cov, err)
output        = get_output(pre, metadata)
print("input shape:", input.shape)
print(input)
print("output shape:", output.shape)
print(output)

print("Getting travaltes")
tra_val_tes   = get_tra_val_tes(metadata["ind_id"].to_numpy(), file = "/home/moicoll/spaceNNtime/sandbox/{sim}/{met}/tra_val_tes_{met}.json".format(sim = sim, met = met))
print("Done")

prev_time = time.time()
carryon_file = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/carryon.txt".format(sim = sim, exp = exp)
start_batch, new_file = carryon(carryon_file)
for j, i in enumerate(range(start_batch, len(tra_val_tes))):
    tf.random.set_seed(1234)
    i = str(i)
    print("Processing batch {}".format(i), flush = True)

    norm_features, mean_features, variance_features = normalizer(nor = nfe, array = input[tra_val_tes[i]["tra"]])
    norm_labels,   mean_labels,   variance_labels   = normalizer(nor = nla, array = output[tra_val_tes[i]["tra"]])
    
    model = spaceNNtime(output_shape  = output.shape[1], 
                        norm          = norm_features, 
                        dropout_prop  = dro, 
                        l             = lay, 
                        n             = nod, 
                        loss_function = los, 
                        w_time        = wti, 
                        w_space       = wsp)

    if j == 0:
        model.summary()

    checkpoint, earlystop, reducelr = callbacks(weights_file_name = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/models/group{i}_weights.hdf5".format(sim = sim, exp = exp, i = i))
    if dat == "default":
        history = train_spaceNNtime(model             = model, 
                                    tra_fea           = input[tra_val_tes[i]["tra"]], 
                                    tra_lab           = norm_labels(output[tra_val_tes[i]["tra"], :]).numpy(), 
                                    val_fea           = input[tra_val_tes[i]["val"]], 
                                    val_lab           = norm_labels(output[tra_val_tes[i]["val"], :]).numpy(),
                                    callbacks         = [checkpoint, earlystop, reducelr],
                                    tra_sample_weight = wsa[tra_val_tes[i]["tra"]])
                                    #val_sample_weight = wsa[tra_val_tes[i]["val"]])

    elif dat == "custom":
        tra_gen = CustomDataGen(x = input[tra_val_tes[i]["tra"]], y = norm_labels(output[tra_val_tes[i]["tra"], :]).numpy(), x_weights = wsa[tra_val_tes[i]["tra"]])
        val_gen = CustomDataGen(x = input[tra_val_tes[i]["val"]], y = norm_labels(output[tra_val_tes[i]["val"], :]).numpy(), x_weights = np.array([]))

        history = train_spaceNNtime_datagen(model             = model, 
                                            tra_gen           = tra_gen, 
                                            val_gen           = val_gen, 
                                            callbacks         = [checkpoint, earlystop, reducelr])

    plot_loss(history = history, fig_path = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/history_plots/group{i}_history.png".format(sim = sim, exp = exp, i = i))

    pred = model.predict(input[tra_val_tes[i]["tes"]])
    pred = (pred*np.sqrt(variance_labels))+mean_labels

    new_file  = write_pred(sim       = sim, 
                           exp       = exp, 
                           nam       = nam, 
                           typ       = typ, 
                           gro       = i, 
                           ind       = metadata["ind_id"].to_numpy()[tra_val_tes[i]["tes"]],
                           idx       = tra_val_tes[i]["tes"], 
                           snp       = input.shape[1], 
                           run       = (time.time()-prev_time)/60.0, 
                           pre       = pre, 
                           true      = output[tra_val_tes[i]["tes"]],
                           pred      = pred, 
                           new_file  = new_file, 
                           file_name = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/pred.txt".format(sim = sim, exp = exp))

    print((time.time()-prev_time)/60.0, "min")
    prev_time = time.time()

    with open(carryon_file, "w") as f:
        f.write("{}".format(i))