ssh nk8155@gadi.nci.org.au
Madh@0417


qsub -I  -l walltime=5:00:00,mem=7GB,ncpus=10,jobfs=5GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23


cd /g/data/jq77/nuwan/MS_with_traditional/
module load python3/3.8.5
. /scratch/jq77/nk8155/env2/bin/activate
python -W ignore run_experiments.py


