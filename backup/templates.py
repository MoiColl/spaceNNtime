#A. Importing
from gwf import Workflow
#import pandas as pd
#import numpy as np
#B. Templates

##B.1.
def spaceNNtime_sim(exp, data_type, norm, ys, p, mean_depth, std_depth, e, width):
	'''
	Runs spaceNNtime in simulations
	'''
	inputs  = ["sandbox/europe/processed_tree/tree.trees",
	           "sandbox/europe/spaceNNtime/exp{exp}/metadata.csv".format(exp = exp),
	           "sandbox/europe/spaceNNtime/exp{exp}/exp{exp}.yaml".format(exp = exp)]
	outputs = ["sandbox/europe/spaceNNtime/exp{exp}/DONE.txt".format(exp = exp)]
	options = {
			'memory'  : '8g',
			'walltime': '1-00:00:00',
			'account' : 'GenerationInterval'
	}
	if exp != "001":
		options['memory'] = "20g"
		#options['queue']  = "gpu"
		#options['gres']   = "gpu:1"
	else:
		options['memory'] = "50g"#"50g"


	spec = '''
	echo "JOBID:" $PBS_JOBID
	
	mkdir -p sandbox/europe/spaceNNtime/exp{exp}/models
	mkdir -p sandbox/europe/spaceNNtime/exp{exp}/history_plots

	python scripts/spaceNNtime_sim.py {exp} {data_type} {norm} {ys} {p} {mean_depth} {std_depth} {e} {width}

	touch sandbox/europe/spaceNNtime/exp{exp}/DONE.txt
	'''.format(exp = exp, data_type = data_type, norm = norm, ys = ys, p = p, mean_depth = mean_depth, std_depth = std_depth, e = e, width = width)


	return inputs, outputs, options, spec