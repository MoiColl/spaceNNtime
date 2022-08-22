#A. Importing
import pandas as pd
import os
from templates import *
gwf = Workflow()


#B. Variables

workingdir  = "/home/moicoll/spaceNNtime/"
experiments = pd.read_csv("{}files/experiments.csv".format(workingdir), delimiter = ";")

#C. CODE
for i in range(len(experiments)):
	dir_meta = "{}data/{}/metadata".format(workingdir, experiments["sim"][i], experiments["met"][i])
	metadata = "{}/{}.txt".format(dir_meta, experiments["met"][i])
	# Creating metadata files for each experiment to downsample from the original simulation
	if not os.path.isfile(metadata):
		if not os.path.isdir(dir_meta):
			os.mkdir(dir_meta)
		if experiments["siz"][i] == "real":
			special_sampling(metadata   = pd.read_csv("{}data/{}/metadata.txt".format(workingdir, experiments["sim"][i]), delimiter = "\t"), 
			                 time_bins  = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,  9000, 10000, 11000, 12000, 13000, 14000, 24000, 25000, 31000, 32000, 33000, 34000, 37000, 45000],
							 n          = [ 370,  369,  161,  135,  177,  148,   86,  105,    40,    31,    24,     3,     1,     2,     1,     1,     1,     2,     1,     1,     2,     1],
			                 step       =  1000,
							 file_name  =  metadata)
		else:
			(pd.read_csv("{}data/{}/metadata.txt".format(workingdir, experiments["sim"][i]), delimiter = "\t")
				.sample(n=int(experiments["siz"][i]), replace=False, weights=None, random_state=1234)
				.to_csv(metadata, sep='\t', header=True, index=False, mode='w'))
	
	# Run spaceNNtime
	gwf.target_from_template("sNNt_{}_{}_{}".format(experiments["sim"][i],       experiments["exp"][i],       experiments["nam"][i]), 
							  spaceNNtime_sim(sim = experiments["sim"][i], exp = experiments["exp"][i], nam = experiments["nam"][i], 
							                  met = experiments["met"][i],
							                  snp = experiments["snp"][i], pre = experiments["pre"][i], typ = experiments["typ"][i], 
											  cov = experiments["cov"][i], std = experiments["std"][i], err = experiments["err"][i], 
											  los = experiments["los"][i], nfe = experiments["nfe"][i], nla = experiments["nla"][i], 
											  wti = experiments["wti"][i], wsp = experiments["wsp"][i], wsa = experiments["wsa"][i], 
											  nod = experiments["nod"][i], 
											  mem = experiments["mem"][i], que = experiments["que"][i], tim = experiments["tim"][i]))