Environment - env

ssh nk8155@gadi.nci.org.au
Madh@0417


qsub -I  -l walltime=10:00:00,mem=15GB,ncpus=8,jobfs=5GB -P qw14 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

qsub -I  -l walltime=15:00:00,mem=20GB,ncpus=8,jobfs=5GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23

cd /g/data/jq77/nuwan/MS_with_traditional/
module load python3/3.8.5
. /scratch/jq77/nk8155/env2/bin/activate
python -W ignore GVFA_STD_298.py --seeds 0,1,2,3,4 --dims 100,500,1000,2000,5000,10000


cd /g/data/jq77/nuwan/MS_with_traditional/
module unload python3
module load python3/3.8.5
. /scratch/jq77/nk8155/env2/bin/activate


python GVFA_STD_298.py --mode gvfa --seeds 0,1,2,3,4 --dims 100,500,1000,2000,5000,10000

python GVFA_STD_298.py --mode combined --seeds 0,1,2,3,4
# or simply:
python GVFA_STD_298.py --seeds 0,1,2,3,4