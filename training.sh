# python3 train.py --network_arch SSP --batch-size 32 --ngf 4 --parallel
# python3 train.py --network_arch NoSSP --batch-size 32 --ngf 4 --parallel
# python3 train.py --network_arch SSP --batch-size 64 --ngf 4 --parallel
# python3 train.py --network_arch NoSSP --batch-size 64 --ngf 4 --parallel
# 
# python3 train.py --network_arch SSP --batch-size 32 --ngf 8 --parallel
# python3 train.py --network_arch NoSSP --batch-size 32 --ngf 8 --parallel
# python3 train.py --network_arch SSP --batch-size 64 --ngf 8 --parallel
# python3 train.py --network_arch NoSSP --batch-size 64 --ngf 8 --parallel
# 
# python3 train.py --network_arch SSP --batch-size 32 --ngf 16 --parallel
# python3 train.py --network_arch NoSSP --batch-size 32 --ngf 16 --parallel
# python3 train.py --network_arch SSP --batch-size 64 --ngf 16 --parallel
# python3 train.py --network_arch NoSSP --batch-size 64 --ngf 16 --parallel
# 
# python3 train.py --network_arch SSP --batch-size 32 --ngf 32 --parallel
# python3 train.py --network_arch NoSSP --batch-size 32 --ngf 32 --parallel
# python3 train.py --network_arch SSP --batch-size 64 --ngf 32 --parallel
# python3 train.py --network_arch NoSSP --batch-size 64 --ngf 32 --parallel
# 
# python3 train.py --network_arch SSP --batch-size 32 --ngf 64 --parallel
# python3 train.py --network_arch NoSSP --batch-size 32 --ngf 64 --parallel
# python3 train.py --network_arch SSP --batch-size 64 --ngf 64 --parallel
# python3 train.py --network_arch NoSSP --batch-size 64 --ngf 64 --parallel


for execut in 1 2 3 4 5
do
    for ngfval in 4 8 16 32 64
    do
        for bsval in 32 64
        do
            python3 train.py --n_execution $execut --network_arch SSP --batch-size $bsval --ngf $ngfval --parallel
            python3 train.py --n_execution $execut --network_arch NoSSP --batch-size $bsval --ngf $ngfval --parallel
        done
    done
done


