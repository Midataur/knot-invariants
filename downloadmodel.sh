rm -r -f ./model_saves/$1_checkpoint
mkdir ./model_saves/$1_checkpoint
rsync -P -e ssh petschackm@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2163/knot-invariants/model_saves/$1_checkpoint/model.safetensors ./model_saves/$1/model.safetensors
rsync -P -e ssh petschackm@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2163/knot-invariants/model_saves/$1_checkpoint/config.pickle ./model_saves/$1/config.pickle