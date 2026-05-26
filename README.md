Environment - env

ssh nk8155@gadi.nci.org.au
Madh@0417


qsub -I  -l walltime=10:00:00,mem=10GB,ncpus=8,jobfs=5GB -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23


cd /g/data/jq77/nuwan/MS_with_traditional/
module load python3/3.8.5
. /scratch/jq77/nk8155/env2/bin/activate
python -W ignore run_experiments.py


cd /g/data/jq77/nuwan/MS_with_traditional/
module load python3/3.8.5
. /scratch/jq77/nk8155/env2/bin/activate


python GVFA_tradition_combined.py --combine concat --regressor xgb  --hv-dims 2000 5000 10000
python GVFA_tradition_combined.py --combine concat_scale_desc  --regressor xgb  --hv-dims 2000 5000 10000
python GVFA_tradition_combined.py --combine concat_scale_both  --regressor xgb  --hv-dims  2000 5000 10000
python GVFA_tradition_combined.py --combine concat_scale_full  --regressor xgb  --hv-dims 2000 5000 10000
python GVFA_tradition_combined.py --combine concat_rp --regressor xgb  --hv-dims 2000 5000 10000
python GVFA_tradition_combined.py --combine concat_pca  --regressor xgb  --hv-dims 2000 5000 10000
python GVFA_tradition_combined.py --combine superposition  --regressor xgb  --hv-dims  2000 5000 10000

--regressor: xgb, svr_rbf, svr_linear, ridge