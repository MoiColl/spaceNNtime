import pandas as pd
import numpy as np

datmet_filter = ["Direct", "Known"]

print("1. getting individual data")
ind      = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.ind", index_col = None, header = None, names = ["indivi", "sexsex", "poppop"]).filter(["indivi"])
filt     = pd.read_table("/home/moicoll/spaceNNtime/files/AADR_filtered_metadata.txt", index_col = None)
metadata = (ind.join(filt.set_index('indivi'), on = "indivi",  how = "inner")
               .reset_index()
               .rename(columns={"latitu" : "lat", "longit" : "lon", "datmea" : "time"})
               .query('datme2 in [{}]'.format(",".join(['"{}"'.format(x) for x in datmet_filter]))))

print(metadata.index.to_numpy())
# print(metadata.query('datme2 in ["Direct", "Known"]')["index"].to_numpy()[:10])

# print("2. getting snp data")
# chromosome = 2
# start = 0
# end   = 10_000_000

# snp     = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.snp", index_col = None, header = None, names = ["snp", "chr", "gen", "pos", "ref", "alt"])
# snp_sta = snp[(snp.chr == chromosome) & (snp.pos >= start) & (snp.pos < end)].index.to_numpy()[0]
# snp_end = snp[(snp.chr == chromosome) & (snp.pos >= start) & (snp.pos < end)].index.to_numpy()[-1]
# snp_idx = np.array([i for i in snp[(snp.chr == chromosome) & (snp.pos >= start) & (snp.pos < end)].index])

# print(snp_sta)

# print("3. getting geno data")
# geno = []
# with open("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public.eigenstratgeno", "r") as f:
#     for i, l in enumerate(f):
#         if i >= snp_idx[0] and i <= snp_idx[-1]:
#             geno.append([int(x) for x in l.strip()])
#         elif i > snp_idx[-1]:
#             break
# geno = np.array(geno)[:, ind_idx]

# print("geno.shape", geno.shape)
# print("ind_idx", ind_idx.shape)
# print("snp_idx", snp_idx.shape)

        

