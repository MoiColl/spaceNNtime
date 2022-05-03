#A. Importing
import pandas as pd

from templates import *
gwf = Workflow()


#B. Code

workingdir  = "/home/moicoll/spaceNNtime/"
experiments = pd.read_csv("{}files/experiments.csv".format(workingdir), delimiter = ";")

for i in len(experiments):
	if experiments["dat"][i]

# #B. Code
# for exp in ["001", "002", "003", "004", "005", "006", "007", "008"]:
# 	gwf.target_from_template("sNNt_{exp}".format(exp = exp), 
# 		                     spaceNNtime_sim(exp = exp,  data_type = "ac", norm = "True", ys = "spaceNNtime", p = 1.0, 
# 							                 mean_depth = None, std_depth = None, e = None,
# 											 width = 256))

# for exp, p in zip(["009", "010", "011", "012"], [0.75, 0.5, 0.25, 0.1]):
# 	gwf.target_from_template("sNNt_{exp}".format(exp = exp), 
# 							 spaceNNtime_sim(exp = exp,  data_type = "ac", norm = "True", ys = "spaceNNtime", p = p, 
# 							                 mean_depth = None, std_depth = None, e = None,
# 											 width = 256))

# for exp, ys in zip(["013", "014"], ["space", "time"]):
# 	gwf.target_from_template("sNNt_{exp}".format(exp = exp), 
# 							 spaceNNtime_sim(exp = exp,  data_type = "ac", norm = "True", ys = ys, p = 1.0, 
# 							                 mean_depth = None, std_depth = None, e = None,
# 											 width = 256))

# for exp, mean_depth in zip(["015", "016", "017"], [5, 15, 30]):
# 	gwf.target_from_template("sNNt_{exp}".format(exp = exp), 
# 							 spaceNNtime_sim(exp = exp,  data_type = "gl", norm = "True", ys = "spaceNNtime", p = 1.0, 
# 							                 mean_depth = mean_depth, std_depth = 5, e = 0.01,
# 											 width = 256))

# for exp, width in zip(["018", "019", "020"], [256/4, 256/2, 256*2]):
# 	width = int(width)
# 	gwf.target_from_template("sNNt_{exp}".format(exp = exp), 
# 							 spaceNNtime_sim(exp = exp,  data_type = "ac", norm = "True", ys = "spaceNNtime", p = 1.0, 
# 							                 mean_depth = None, std_depth = None, e = None,
# 											 width = width))