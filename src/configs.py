# Dataset setting
TRAIN_SET_PATH = "./data/ua.base"
TEST_SET_PATH = "./data/ua.test"
TITLE_FEATURE_PATH = "./data/BERT_features"
VALIDATION = 0.2

# Training params
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 80

# NEPTUNE SETTING
NEPTUNE_NAME = "Baseline Model with Embedding and title feature(BERT)"
DESCRIPTION = """
    A simple baseline model. 
    Add NDCG metric for validation. 
    Use learnt embedding for categorical feature instead of one-hot encoding.
    Add title feature extracted by BERT model.
"""
MODE = "debug"

# MODEL CONFIG
EMBEDDING_SIZE = 32