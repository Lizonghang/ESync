# for train
BATCH_SIZE = 64
NETWORK = "resnet18-v1"
MODE = "esync"
SPLIT_BY_CLASS = False
DATA_DIR = "/home/lizh/ESync/data"

# for evaluation
EVAL_DURATION = 1
LOG_DIR = "/home/lizh/ESync/logs"

# for optimizer
LEARNING_RATE = 1e-3
LEARNING_RATE_LOCAL = LEARNING_RATE
LEARNING_RATE_GLOBAL = 1.0

# devices
USE_CPU = False
DEFAULT_GPU_ID = 0

# state server
STATE_SERVER_IP = "10.1.1.34"
STATE_SERVER_PORT = 10010
COMMON_URL = "http://{ip}:{port}/%s/".format(ip=STATE_SERVER_IP, port=STATE_SERVER_PORT)
