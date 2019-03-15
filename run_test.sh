export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
config=./Config/config-fusion.cfg
device=cuda:0
log_name=log
# device ["cpu", "cuda:0", "cuda:1", ......]
nohup python -u main.py --config $config --device $device --t_data test --test > $log_name 2>&1 &
tail -f $log_name
 


