import mixup2_app

ENCODER_ID = 'roberta-base' # id from HuggingFace
MODEL_REFERENCE = 'RoBERTa'
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 768
BATCH_SIZE = 16
DROPOUT_RATE = 0.2

MIXUP_ALPHA = 0.1
AUGMENTATION_RATE = 0.5  # augmentation rate for pointed classes
CLASSES_TO_AUGMENT = ['Argument', 'Statute', 'Precedent', 'RulingByLowerCourt', 'RulingByPresentCourt']

N_EPOCHS = 4
LEARNING_RATE = 1e-5

train_params = {}
train_params['encoder_id'] = ENCODER_ID
train_params['model_reference'] = MODEL_REFERENCE
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['batch_size'] = BATCH_SIZE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['n_epochs'] = N_EPOCHS
train_params['learning_rate'] = LEARNING_RATE
train_params['mixup_alpha'] = MIXUP_ALPHA
train_params['augmentation_rate'] = AUGMENTATION_RATE
train_params['classes_to_augment'] = CLASSES_TO_AUGMENT
train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8

#train_params['n_documents'] = 1
train_params['n_iterations'] = 5
train_params['use_mock'] = False

mixup2_app.evaluate_BERT(train_params)