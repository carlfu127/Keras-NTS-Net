gpus = '0,1,2,3'
batch_size = 16
num_cls = 200
INPUT_SIZE = (448, 448)
PROPOSAL_NUM = 6
STEPS = [60, 100]
CAT_NUM = 4
initial_learning_rate = 0.001
weight_decay = 1e-4
save_weights_path = 'models/cub200'
dataset_dir = 'datasets/CUB_200_2011'
