mkdir -p ~/data/
scp qxz396@racimocomp04fl:/projects/racimolab/people/qxz396/spaceNNtime/data/europe/tree.trees ~/data/.
scp qxz396@racimocomp04fl:/projects/racimolab/people/qxz396/spaceNNtime/data/europe/metadata.txt ~/data/.

scp ~/data/tree.trees moicoll@login.genome.au.dk:/home/moicoll/faststorage/spaceNNtime/data/europe/.
scp ~/data/metadata.txt moicoll@login.genome.au.dk:/home/moicoll/faststorage/spaceNNtime/data/europe/.
rm -r ~/data