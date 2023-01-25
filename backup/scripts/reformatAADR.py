import pandas as pd

ind = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.ind", index_col = None, header = None, names = ["ind", "sex", "pop"])

snp = pd.read_table("/home/moicoll/spaceNNtime/data/AADR/v54.1_1240K_public_nospaces.snp", sep = "\t", index_col = None, header = None, names = ["snp", "chr", "gen", "pos", "ref", "alt"])

print(ind)
print(snp)