import sys
import tskit
import pandas as pd
import time
sys.path.append('scripts/')
from spaceNNtime_templates import *
from tensorflow.python.client import device_lib

print("Which devices are available?")
print(device_lib.list_local_devices())

sim, exp, nam, met, snp, pre, typ, cov, err, los, nfe, nla, wti, wsp, wsa, nod  = sys.argv[1:]
snp = float(snp)
cov = float(cov)
err = float(err)
nod = int(nod)
wti = float(wti)
wsp = float(wsp)

ts            = tskit.load("/home/moicoll/spaceNNtime/data/{sim}/tree.trees".format(sim = sim))
metadata      = pd.read_csv("/home/moicoll/spaceNNtime/data/{sim}/metadata/{met}.txt".format(sim = sim, met = met), delimiter = "\t")
input         = get_input(ts, metadata, snp, typ, cov, err)
output        = get_output(pre, metadata)
print("input shape:", input.shape)
print(input)
print("output shape:", output.shape)
print(output)

if wsa == "None":
    wsa = np.array(1)
if wsa == "coverage":
    pass

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

    norm_features = normalizer(nor = nfe, input_shape = input.T.shape[1])
    if nfe != "None":
        norm_features.adapt(input[:, tra_val_tes[i]["tra"]].T)
    norm_labels   = normalizer(nor = nla, input_shape = output.shape[1])
    if nla != "None":
        norm_labels.adapt(output[tra_val_tes[i]["tra"], :])

    model = spaceNNtime(output_shape  = output.shape[1], 
                        norm          = norm_features, 
                        dropout_prop  = 0.25, 
                        l             = 10, 
                        n             = nod, 
                        loss_function = los, 
                        w_time        = wti, 
                        w_space       = wsp, 
                        w_sample      = wsa)

    if j == 0:
        model.summary()

    checkpoint, earlystop, reducelr = callbacks(weights_file_name = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/models/group{i}_weights.hdf5".format(sim = sim, exp = exp, i = i))

    history = train_spaceNNtime(model     = model, 
                                tra_fea   = input[:, tra_val_tes[i]["tra"]].T, 
                                tra_lab   = norm_labels(output[tra_val_tes[i]["tra"], :]), 
                                val_fea   = input[:, tra_val_tes[i]["val"]].T, 
                                val_lab   = norm_labels(output[tra_val_tes[i]["val"], :]),
                                callbacks = [checkpoint, earlystop, reducelr])

    plot_loss(history = history, fig_path = "/home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/history_plots/group{i}_history.png".format(sim = sim, exp = exp, i = i))

    pred = model.predict(input[:, tra_val_tes[i]["tes"]].T)
    
    new_file  = write_pred(sim       = sim, 
                           exp       = exp, 
                           nam       = nam, 
                           typ       = typ, 
                           gro       = i, 
                           ind       = metadata["ind_id"].to_numpy()[tra_val_tes[i]["tes"]],
                           idx       = tra_val_tes[i]["tes"], 
                           snp       = input.shape[0], 
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