import sys
import tskit
import pandas as pd
import time
sys.path.append('scripts/')
from spaceNNtime_templates import *
from tensorflow.python.client import device_lib

print("Which devices are available?")
print(device_lib.list_local_devices())

exp, nam, met, cro, sta, end, dmt, pre, lay, dro, typ, los, nfe, nla, wti, wsp, wsa, nod, dat  = sys.argv[1:]
crl = [int(x) for x in cro.split("t")]
if len(crl) > 1:
    crl = [x for x in range(crl[0], crl[1]+1)]
sta = int(int(sta)*1e6)
end = int(int(end)*1e6)
lay = int(lay)
dro = float(dro)
nod = int(nod)
wti = float(wti)
wsp = float(wsp)

ind      = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.ind", index_col = None, header = None, names = ["indivi", "sexsex", "poppop"]).filter(["indivi"])
filt     = pd.read_table("/home/moicoll/spaceNNtime/files/AADR_filtered_metadata.txt", index_col = None, na_values="..")
print("creating new metadata")
print(dmt)
print(dmt.replace("_", " ").replace('\\', ''))
metadata = (ind.join(filt.set_index('indivi'), on = "indivi",  how = "inner")
               .reset_index()
               .rename(columns={"latitu" : "lat", "longit" : "lon", "datmea" : "time"})
               .query(dmt.replace("_", " ").replace('\\', '')))

print(metadata)


print("Reading input...")
input, snp    = get_input_AADR(metadata, crl, sta, end)
if type(input) != type(None):
        
    print("Reading output...")
    output        = get_output(pre, metadata)

    print("input shape:", input.shape)
    print(input)
    print("output shape:", output.shape)
    print(output)

    write_qc_ind(metadata.indivi.to_numpy(), input, "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/qc_ind_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end))
    write_qc_snp(snp, input, "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/qc_snp_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end))

    if wsa == "None":
        wsa = np.ones(metadata.shape[0])
    elif wsa == "coverage":
        qc_ind = pd.read_table("/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/qc_ind_{cro}_{sta}_{end}.txt".format(exp = exp, cro = cro, sta = sta, end = end), names = ["indivi", "noncal", "homref", "hethet", "homalt", "noncha"])
        qc_ind["callab"] = (qc_ind["homref"]+qc_ind["hethet"]+qc_ind["homalt"])/(qc_ind["homref"]+qc_ind["hethet"]+qc_ind["homalt"]+qc_ind["noncal"])
        wsa    = metadata.join(qc_ind.filter(["indivi", "callab"]).set_index('indivi'), on = "indivi")["callab"].to_numpy()
    elif "timing" in wsa:
        timing_val = int(wsa[len("timing"):])
        wsa = np.ones(metadata.shape[0])
        wsa[metadata["datmet"] != "Context"] = timing_val 
    elif wsa == "datstd":
        wsa = (1-(metadata["datstd"].to_numpy()/metadata["time"].to_numpy()))
        print(wsa)
        print(wsa.shape)
    else:
        sys.exit("No correct wsa variable")

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

        checkpoint, earlystop, reducelr = callbacks(weights_file_name = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/models/group{i}_{cro}_{sta}_{end}_weights.hdf5".format(exp = exp, i = i, cro = cro, sta = sta, end = end))
        if dat == "default":
            history = train_spaceNNtime(model             = model, 
                                        tra_fea           = input[tra_val_tes[i]["tra"]], 
                                        tra_lab           = norm_labels(output[tra_val_tes[i]["tra"], :]), 
                                        val_fea           = input[tra_val_tes[i]["val"]], 
                                        val_lab           = norm_labels(output[tra_val_tes[i]["val"], :]),
                                        callbacks         = [checkpoint, earlystop, reducelr],
                                        tra_sample_weight = wsa[tra_val_tes[i]["tra"]])
                                        #val_sample_weight = wsa[tra_val_tes[i]["val"]])
        elif dat == "custom":
            tra_gen = CustomDataGen(x = input[tra_val_tes[i]["tra"]], y = norm_labels(output[tra_val_tes[i]["tra"]]).numpy(), x_weights = wsa[tra_val_tes[i]["tra"]])
            val_gen = CustomDataGen(x = input[tra_val_tes[i]["val"]], y = norm_labels(output[tra_val_tes[i]["val"]]).numpy(), x_weights = np.array([]))

            history = train_spaceNNtime_datagen(model             = model, 
                                                tra_gen           = tra_gen, 
                                                val_gen           = val_gen, 
                                                callbacks         = [checkpoint, earlystop, reducelr])
        
        
        if dat == "default":
            plot_loss(history = history, fig_path = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/history_plots/group{i}_{cro}_{sta}_{end}_history.png".format(exp = exp, i = i, cro = cro, sta = sta, end = end))
        elif dat == "custom":
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.to_csv("/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/history_plots/group{i}_{cro}_{sta}_{end}_history_rawdata.csv".format(exp = exp, i = i, cro = cro, sta = sta, end = end), 
                        mode='w', header=True, sep = "\t", index = False)
            plot_loss(history = history, fig_path = "/home/moicoll/spaceNNtime/sandbox/AADR/{exp}/history_plots/group{i}_{cro}_{sta}_{end}_history.png".format(exp = exp, i = i, cro = cro, sta = sta, end = end))


        pred = model.predict(input[tra_val_tes[i]["tes"]])
        pred = (pred*np.sqrt(variance_labels))+mean_labels

        new_file  = write_pred_AADR(sim       = "AADR", 
                                    exp       = exp, 
                                    nam       = nam, 
                                    typ       = typ,
                                    cro       = cro,
                                    sta       = sta, 
                                    end       = end,
                                    gro       = i, 
                                    ind       = metadata["indivi"].to_numpy()[tra_val_tes[i]["tes"]],
                                    idx       = tra_val_tes[i]["tes"], 
                                    snp       = input.shape[1], 
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
else:
    print("No SNPs for this window...")
