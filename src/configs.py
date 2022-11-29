# Dataset setting
TRAIN_SET_PATH = "./data/ua.base"
TEST_SET_PATH = "./data/ua.test"
VALIDATION = 0.2

# Training params
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 80

# NEPTUNE SETTING
NEPTUNE_NAME = "Baseline Model"
DESCRIPTION = """
    A simple baseline model. Add NDCG metric for validation.
"""
MODE = "async"
