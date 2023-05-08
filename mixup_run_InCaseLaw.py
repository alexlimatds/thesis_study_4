import mixup_app

ENCODER_ID = 'law-ai/InCaseLawBERT' # id from HuggingFace
MODEL_REFERENCE = 'InCaseLaw'
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 768
BATCH_SIZE = 16
DROPOUT_RATE = 0.2

DATASET = 'facts' # 'malik' or 'facts'
MIXUP_ALPHA = 1.0
AUGMENTATION_RATE = 5.0  # augmentation rate for pointed classes
#CLASSES_TO_AUGMENT = ['Precedent', 'RulingByLowerCourt'] # malik dataset
#CLASSES_TO_AUGMENT = ['Fact', 'RulingByPresentCourt', 'Other'] # facts dataset
CLASSES_TO_AUGMENT = ['Fact', 'RulingByPresentCourt'] # facts dataset

N_EPOCHS_ENCODER = 4
STOP_EPOCH_ENCODER = 2
LEARNING_RATE_ENCODER = 1e-5

N_EPOCHS_CLASSIFIER = 200
LEARNING_RATE_CLASSIFIER = 1e-2

train_params = {}
train_params['encoder_id'] = ENCODER_ID
train_params['model_reference'] = MODEL_REFERENCE
train_params['dataset'] = DATASET
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['batch_size'] = BATCH_SIZE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['n_epochs_encoder'] = N_EPOCHS_ENCODER
train_params['stop_epoch_encoder'] = STOP_EPOCH_ENCODER
train_params['learning_rate_encoder'] = LEARNING_RATE_ENCODER
train_params['n_epochs_classifier'] = N_EPOCHS_CLASSIFIER
train_params['learning_rate_classifier'] = LEARNING_RATE_CLASSIFIER
train_params['mixup_alpha'] = MIXUP_ALPHA
train_params['augmentation_rate'] = AUGMENTATION_RATE
train_params['classes_to_augment'] = CLASSES_TO_AUGMENT
train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8

#train_params['n_documents'] = 1
train_params['n_iterations'] = 5
train_params['use_mock'] = False

mixup_app.evaluate_BERT(train_params)
