rm -r -f ./model_saves/$1
mkdir ./model_saves/$1
scp petschackm@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2163/braids/model_saves/$1/model.safetensors ./model_saves/$1/model.safetensors
scp petschackm@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2163/braids/model_saves/$1/config.pickle ./model_saves/$1/config.pickle