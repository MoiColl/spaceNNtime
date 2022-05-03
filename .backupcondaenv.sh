pref=`date +"%y_%m_%d-%H_%M_%S"`
envn=`echo $CONDA_DEFAULT_ENV`
if test -f "environments/environment-${envn}.yml"; 
then
    echo "The file environments/environment-${envn}.yml exists. Moving the existing one to environments/old/environment-${envn}-${pref}.yml and creating a new one."
    mkdir -p environments/old
    mv environments/environment-${envn}.yml environments/old/environment-${envn}-${pref}.yml
else
	echo "The file environment-${envn}.yml does not exists. Creating one."
fi
time conda env export > environments/environment-${envn}.yml
echo "done"
