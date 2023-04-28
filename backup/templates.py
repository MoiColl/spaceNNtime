#A. Importing
from gwf import Workflow

#B. Templates
##B.1.
def all_done_windows(inp, exp, cro, sta, end, win):
	'''
	A dummy job to just flag that the same job for different inputs have been done
	'''
	sta = int(sta*1e6)
	end = int(end*1e6)
	win = int(win*1e6)
	inputs  = inp
	outputs = ["/home/moicoll/spaceNNtime/sandbox/completed/AADR_{exp}_{cro}_{sta}_{end}_{win}.DONE".format(exp = exp, cro = cro, sta = int(sta/1e6), end = int(end/1e6), win = int(win/1e6))]
	options = {'memory' : '1g', 'walltime': '01:00:00', 'account' : 'GenerationInterval'}
	spec    = '''
	source /home/moicoll/.bash_profile
	conda activate sNNt_au
	
	echo "JOBID            : " $PBS_JOBID
	echo "HOSTNAME         : " $HOSTNAME
	echo "CONDA_DEFAULT_ENV: " $CONDA_DEFAULT_ENV
	
	working_dir="/home/moicoll/spaceNNtime/sandbox/AADR/{exp}"

	head -n 1 ${{working_dir}}/pred_{cro}_{sta}_$(({sta}+{win})).txt > ${{working_dir}}/pred_{cro}_{sta}_{end}_{win}.txt

	for s in `seq {sta} {win} $(({end}-{win}))`;
	do
		awk '{{if(NR > 1){{print}}}}' ${{working_dir}}/pred_{cro}_${{s}}_$((s+{win})).txt >> ${{working_dir}}/pred_{cro}_{sta}_{end}_{win}.txt
		sleep 1
		rm ${{working_dir}}/pred_{cro}_${{s}}_$((s+{win})).txt
	done

	touch /home/moicoll/spaceNNtime/sandbox/completed/AADR_{exp}_{cro}_$(({sta}/1000000))_$(({end}/1000000))_$(({win}/1000000)).DONE

	echo ""
	echo ""
	echo ""
	echo "JOBINFO"
	echo "======="
	jobinfo $PBS_JOBID
	'''.format(exp = exp, cro = cro, sta = sta, end = end, win = win)
	return inputs, outputs, options, spec

##B.2.
def spaceNNtime_sim(sim, exp, nam, met, snp, pre, lay, dro, typ, cov, std, err, los, nfe, nla, wti, wsp, wsa, nod, mem, que, tim):
	'''
	Runs spaceNNtime for simulated data
	'''
	inputs  = ["/home/moicoll/spaceNNtime/data/{sim}/tree.trees".format(sim = sim),
	           "/home/moicoll/spaceNNtime/data/{sim}/metadata/{met}.txt".format(sim = sim, met = met)]
	outputs = ["/home/moicoll/spaceNNtime/sandbox/completed/{sim}_{exp}.DONE".format(sim = sim, exp = exp)]
	options = {'memory'  : mem, 'walltime': tim, 'account' : 'GenerationInterval'}
	if que == "gpu":
		options['queue']  = "gpu"
		options['gres']   = "gpu:1"
	spec = '''
	source /home/moicoll/.bash_profile
	conda activate sNNt_au
	
	echo "JOBID            : " $PBS_JOBID
	echo "HOSTNAME         : " $HOSTNAME
	echo "CONDA_DEFAULT_ENV: " $CONDA_DEFAULT_ENV
	
	mkdir -p /home/moicoll/spaceNNtime/sandbox/completed
	mkdir -p /home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/models
	mkdir -p /home/moicoll/spaceNNtime/sandbox/{sim}/{exp}/history_plots

	python /home/moicoll/spaceNNtime/scripts/spaceNNtime_sim.py {sim} {exp} {nam} {met} {snp} {pre} {lay} {dro} {typ} {cov} {std} {err} {los} {nfe} {nla} {wti} {wsp} {wsa} {nod}

	touch /home/moicoll/spaceNNtime/sandbox/completed/{sim}_{exp}.DONE
	echo ""
	echo ""
	echo ""
	echo "JOBINFO"
	echo "======="
	jobinfo $PBS_JOBID
	'''.format(sim = sim, exp = exp, nam = nam, met = met, snp = snp, pre = pre, lay = lay, dro = dro, typ = typ, cov = cov, 
	           std = std, err = err, los = los, nfe = nfe, nla = nla, wti = wti, wsp = wsp, wsa = wsa, nod = nod)


	return inputs, outputs, options, spec

##B.3.
def spaceNNtime_AADR(exp, nam, met, cro, sta, end, dmt, pre, lay, dro, typ, los, nfe, nla, wti, wsp, wsa, nod, mem, que, tim):
	'''
	Runs spaceNNtime for simulated data
	'''
	inputs  = []
	outputs = ["/home/moicoll/spaceNNtime/sandbox/completed/AADR_{exp}_{cro}_{sta}_{end}.DONE".format(exp = exp, cro = cro, sta = sta, end = end)]
	options = {'memory'  : mem, 'walltime': tim, 'account' : 'GenerationInterval'}
	if que == "gpu":
		options['queue']  = "gpu"
		options['gres']   = "gpu:1"
	spec = '''
	source /home/moicoll/.bash_profile
	conda activate sNNt_au
	
	echo "JOBID            : " $PBS_JOBID
	echo "HOSTNAME         : " $HOSTNAME
	echo "CONDA_DEFAULT_ENV: " $CONDA_DEFAULT_ENV
	
	mkdir -p /home/moicoll/spaceNNtime/sandbox/completed
	mkdir -p /home/moicoll/spaceNNtime/sandbox/AADR/{exp}/models
	mkdir -p /home/moicoll/spaceNNtime/sandbox/AADR/{exp}/history_plots

	python /home/moicoll/spaceNNtime/scripts/spaceNNtime_AADR.py {exp} {nam} {met} {cro} {sta} {end} {dmt} {pre} {lay} {dro} {typ} {los} {nfe} {nla} {wti} {wsp} {wsa} {nod}

	touch /home/moicoll/spaceNNtime/sandbox/completed/AADR_{exp}_{cro}_{sta}_{end}.DONE
	echo ""
	echo ""
	echo ""
	echo "JOBINFO"
	echo "======="
	jobinfo $PBS_JOBID
	'''.format(exp = exp, nam = nam, met = met, pre = pre, lay = lay, dro = dro, typ = typ, cro = cro, sta = sta, 
	           end = end, dmt = dmt, los = los, nfe = nfe, nla = nla, wti = wti, wsp = wsp, wsa = wsa, nod = nod)


	return inputs, outputs, options, spec


#C. Functions
def special_sampling(metadata, time_bins, n, step, file_name):
	for i in range(len(time_bins)):
		metadata_sub = metadata[(metadata.time >= time_bins[i]-step) & (metadata.time < time_bins[i])]
		if len(metadata_sub) > n[i]:
			(metadata_sub.sample(n=n[i], replace=False, weights=None, random_state=1234)
				.to_csv(file_name, sep='\t', header=True, index=False, mode='a'))
		else:
			metadata_sub.to_csv(file_name, sep='\t', header=True, index=False, mode='a')