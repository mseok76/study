python3 train.py --activation ReLU --lr 0.01 > ./results/ReLU.txt
python3 train.py --activation ReLU --lr 0.001 >> ./results/ReLU.txt
python3 train.py --activation ReLU --lr 0.0001 >> ./results/ReLU.txt

python3 train.py --activation SiLU --lr 0.01 > ./results/SiLU.txt
python3 train.py --activation SiLU --lr 0.001 >> ./results/SiLU.txt
python3 train.py --activation SiLU --lr 0.0001 >> ./results/SiLU.txt

python3 train.py --activation ELU --lr 0.01 > ./results/ELU.txt
python3 train.py --activation ELU --lr 0.001 >> ./results/ELU.txt
python3 train.py --activation ELU --lr 0.0001 >> ./results/ELU.txt

python3 train.py --activation Tanh --lr 0.01 > ./results/Tanh.txt
python3 train.py --activation Tanh --lr 0.001 >> ./results/Tanh.txt
python3 train.py --activation Tanh --lr 0.0001 >> ./results/Tanh.txt
