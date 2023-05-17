import hierarchical_app

ENCODER_ID = 'law-ai/InCaseLawBERT' # id from HuggingFace
MODEL_REFERENCE = 'InCaseLaw'
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 768  # hidden dim of transformer encoder
BATCH_SIZE = 16
DROPOUT_RATE = 0.2

DATASET = '4_roles' # '7_roles' or '4_roles'

N_EPOCHS_ENCODER = 4
STOP_EPOCH_ENCODER = 2
LEARNING_RATE_ENCODER = 1e-5

LSTM_HIDDEN_DIM = 100
N_EPOCHS_CLASSIFIER = 50
LEARNING_RATE_CLASSIFIER = 3e-4

train_params = {}
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['batch_size'] = BATCH_SIZE
train_params['encoder_id'] = ENCODER_ID
train_params['dataset'] = DATASET
train_params['model_reference'] = MODEL_REFERENCE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['n_epochs_encoder'] = N_EPOCHS_ENCODER
train_params['stop_epoch_encoder'] = STOP_EPOCH_ENCODER
train_params['learning_rate_encoder'] = LEARNING_RATE_ENCODER
train_params['n_epochs_classifier'] = N_EPOCHS_CLASSIFIER
train_params['learning_rate_classifier'] = LEARNING_RATE_CLASSIFIER
train_params['lstm_hidden_dim'] = LSTM_HIDDEN_DIM
train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8

#train_params['n_documents'] = 2
train_params['use_dev_set'] = True
#train_params['n_iterations'] = 5
train_params['n_iterations'] = 3
train_params['use_mock'] = False

hierarchical_app.evaluate_BERT(train_params)
