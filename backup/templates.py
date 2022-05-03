#A. Importing
from gwf import Workflow

#B. Templates
##B.1.
def spaceNNtime_sim(sim, exp, nam, met, siz, snp, pre, typ, cov, err, nod, mem, que, tim):
	'''
	Runs spaceNNtime for simulated data
	'''
	inputs  = ["data/{sim}/tree.trees".format(sim = sim),
	           "data/{sim}/downsample/{met}.txt".format(sim = sim, met = met)]
	outputs = ["sandbox/completed/{sim}_{exp}.DONE".format(exp = exp)]
	options = {'memory'  : mem, 'walltime': tim, 'account' : 'GenerationInterval'}
	if que != "gpu":
		options['queue']  = "gpu"
		options['gres']   = "gpu:1"
	spec = '''
	echo "JOBID:" $PBS_JOBID
	
	mkdir -p sandbox/completed
	mkdir -p sandbox/{sim}/{exp}/models
	mkdir -p sandbox/{sim}/{exp}/history_plots

	python scripts/spaceNNtime_sim.py {sim} {exp} {nam} {met} {siz} {snp} {pre} {typ} {cov} {err} {nod}

	touch sandbox/completed/{sim}_{exp}.DONE
	'''.format(sim = sim, exp = exp, nam = nam, met = met, siz = siz, snp = snp, pre = pre, typ = typ, cov = cov, err = err, nod = nod)


	return inputs, outputs, options, spec

#C. Functions
def special_sampling(metadata, time_bins, n, step, downsample):
	for i in range(len(time_bins)):
		metadata_sub = metadata[(metadata.time >= time_bins[i]-step) & (metadata.time < time_bins[i])]
		if len(metadata_sub) > n[i]:
			(metadata_sub.sample(n=n[i], replace=False, weights=None, random_state=1234)
				.to_csv(downsample, sep='\t', header=True, index=False, mode='a'))
		else:
			metadata_sub.to_csv(downsample, sep='\t', header=True, index=False, mode='a')