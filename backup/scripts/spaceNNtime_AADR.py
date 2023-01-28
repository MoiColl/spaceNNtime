import sys
import tskit
import pandas as pd
import time
sys.path.append('scripts/')
from spaceNNtime_templates import *
from tensorflow.python.client import device_lib

print("Which devices are available?")
print(device_lib.list_local_devices())

exp, nam, met, cro, sta, end, dmt, pre, typ, los, nfe, nla, wti, wsp, wsa, nod  = sys.argv[1:]
cro = int(cro)
sta = int(sta)
end   = int(end)
nod   = int(nod)
wti   = float(wti)
wsp   = float(wsp)



ind      = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.ind", index_col = None, header = None, names = ["indivi", "sexsex", "poppop"]).filter(["indivi"])
filt     = pd.read_table("/home/moicoll/spaceNNtime/files/AADR_filtered_metadata.txt", index_col = None)
metadata = (ind.join(filt.set_index('indivi'), on = "indivi",  how = "inner")
               .reset_index()
               .rename(columns={"latitu" : "lat", "longit" : "lon", "datmea" : "time"})
               .query('datme2 in [{}]'.format(",".join(['"{}"'.format(x) for x in dmt.split(",")]))))

if wsa == "None":
    wsa = np.ones(metadata.shape[0])
elif wsa == "coverage":
    sys.exit("No correct wsa variable")

input, snp    = get_input_AADR(metadata, cro, sta, end)
output        = get_output(pre, metadata)

print("input shape:", input.shape)
print(input)
print("output shape:", output.shape)
print(output)

write_qc_ind(metadata.indivi.to_numpy(), input, "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/qc_ind_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end))
write_qc_snp(snp, input, "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/qc_snp_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end))

print("Getting travaltes")
tra_val_tes   = get_tra_val_tes(metadata["indivi"].to_numpy(), file = "/home/moicoll/spaceNNtime/sandbox/AADR/{met}/tra_val_tes_{met}.json".format(met = met))
print("Done")

prev_time = time.time()
carryon_file = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/carryon_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end)
sta_batch, new_file = carryon(carryon_file)
for j, i in enumerate(range(sta_batch, len(tra_val_tes))):
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
                        w_space       = wsp)

    if j == 0:
        model.summary()

    checkpoint, earlystop, reducelr = callbacks(weights_file_name = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/models/group{i}_{cro}_{sta}_{end}_weights.hdf5".format(exp = exp, i = i, cro = cro, sta = sta, end = end))

    history = train_spaceNNtime(model         = model, 
                                tra_fea       = input[:, tra_val_tes[i]["tra"]].T, 
                                tra_lab       = norm_labels(output[tra_val_tes[i]["tra"], :]), 
                                val_fea       = input[:, tra_val_tes[i]["val"]].T, 
                                val_lab       = norm_labels(output[tra_val_tes[i]["val"], :]),
                                callbacks     = [checkpoint, earlystop, reducelr],
                                sample_weight = wsa[tra_val_tes[i]["tra"]])

    plot_loss(history = history, fig_path = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/history_plots/group{i}_{cro}_{sta}_{end}_history.png".format(exp = exp, i = i, cro = cro, sta = sta, end = end))

    pred = model.predict(input[:, tra_val_tes[i]["tes"]].T)
    
    new_file  = write_pred(sim       = "AADR", 
                           exp       = exp, 
                           nam       = nam, 
                           typ       = typ, 
                           gro       = i, 
                           ind       = metadata["indivi"].to_numpy()[tra_val_tes[i]["tes"]],
                           idx       = tra_val_tes[i]["tes"], 
                           snp       = input.shape[0], 
                           run       = (time.time()-prev_time)/60.0, 
                           pre       = pre, 
                           true      = output[tra_val_tes[i]["tes"]],
                           pred      = pred, 
                           new_file  = new_file, 
                           file_name = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/pred_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end))

    print((time.time()-prev_time)/60.0, "min")
    prev_time = time.time()

    with open(carryon_file, "w") as f:
        f.write("{}".format(i))