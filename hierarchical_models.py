# Models and related classes and functions
import torch, transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time, math, singlesc_models

class MockEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MockEncoder, self).__init__()
        self.embedding_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        mock_data = torch.rand((batch_size, self.embedding_dim), device=input_ids.device)

        return mock_data

class HierarchicalSC(torch.nn.Module):
    """
    Sentence Classifier that exploits a transformer model and a BiLSTM network to encode sentences. 
    The transformer is the first-level encoder and it encodes sentences in isolation.
    The BiLSTM is the second-level encoder and it encodes sentences by taking in account all sentences 
    in a document.
    During a training, the transformer does not have his weights updated.
    """
    def __init__(self, single_sent_encoder, n_classes, dropout_rate, embedding_dim, lstm_hidden_dim):
        '''
        Arguments:
            single_sent_encoder: an instance of singlesc_models.SingleSentenceEncoder_BERT.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification ans BiLSTM layers.
            embedding_dim: dimension of hidden units of the BERT kind encoder (e.g., 768 for BERT).
            lstm_hidden_dim: hidden dim of the BiLSTM network.
        '''
        super(HierarchicalSC, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.lstm_hidden_dim = lstm_hidden_dim
        # 1st level encoder
        self.single_sent_encoder = single_sent_encoder
        # 2sd elevel encoder
        self.bilstm = torch.nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=lstm_hidden_dim // 2, 
            batch_first=True, 
            bidirectional=True
        )
        # classifier
        dense_out = torch.nn.Linear(lstm_hidden_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(
            self.dropout, dense_out
        )
    
    def encode_batch(self, input_ids, attention_mask):
        '''
        Performs first-level encoding of a batch of sentences (batch_size = n_sentences).
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
        Returns:
            embeddings : tensor of shape (batch_size, embedding_dim). One embedding for each sentence in batch.
        '''
        self.single_sent_encoder.eval()
        with torch.no_grad():
            cls_embeddings = self.single_sent_encoder(
                input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
                attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
            )
        return cls_embeddings # cls_embeddings.shape: (batch_size, embedding_dim)
    
    def forward(self, sent_embeddings):
        '''
        Each call to this method process a batch of documents.
        The sentence embeddings are fed to a BiLSTM network to get embeddings enriched with document context.
        This method returns one logit tensor for each sentence in the batch.
        Arguments:
            sent_embeddings : The batch of documents as a list of tensors. The list has length equals to batch_size (n docs in batch). 
                              Each tensor has shape (n sentences in doc, embedding_dim).
        Returns:
            logits : tensor of shape (n sentences in batch, n_classes).
        '''
        # 2nd level encoding
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(sent_embeddings, batch_first=True) # padded_embeddings.shape: (batch_size, max_n_sentences, embbeding_dim)
        packed_embs = torch.nn.utils.rnn.pack_sequence(padded_embeddings, enforce_sorted=False)
        packed_lstm_embeddings, _ = self.bilstm(packed_embs)
        lstm_embeddings, lens_lstm_embeddings = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_embeddings, batch_first=True)
        # lstm_embeddings.shape: (batch_size, max_n_sentences, embedding_dim)
        # lens_lstm_embeddings.shape: (batch_size, n_valid_sentences)
        valid_embeddings = []
        for idx_batch in range(lens_lstm_embeddings.shape[0]):
            n_valid = lens_lstm_embeddings[idx_batch]
            valid_embeddings.append(lstm_embeddings[idx_batch, 0:n_valid])
        valid_embeddings = torch.vstack(valid_embeddings) # valid_embeddings.shape: (n valid sentences in batch, lstm_hidden_dim)
        
        # classification
        logits = self.classifier(valid_embeddings)   # logits.shape: (n_sentences, n_classes)

        return logits

class TensorList_Dataset(torch.utils.data.Dataset):
    def __init__(self, sent_embeddings, targets):
        """
        Arguments:
            sent_embeddings: list of tensors with shape (n_sentences, embedding_dim). The items in the list may have different values of n_sentences.
            targets: list of tensors with shape (n_sentences, 1). The items in the list may have different values of n_sentences.
        """
        for t1, t2 in zip(sent_embeddings, targets):
            assert t1.shape[0] == t2.shape[0]
        self.sent_embeddings = sent_embeddings
        self.targets = targets

    def __getitem__(self, index):
        return self.sent_embeddings[index], self.targets[index]
    
    def __len__(self):
        return len(self.targets)

def collate_batch_TensorList(batch):
    '''
    Prepares a batch of TensorList_Dataset items.
    Arguments:
        batch: list of tuples from an instance of TensorList_Dataset.
    Returns:
        A list with the sentence embeddings.
        A list with the targets.
    '''
    embeddings = []
    targets = []
    for e, t in batch:
        embeddings.append(e)
        targets.append(t)
    return embeddings, targets

def first_encoding(ds_list, sentence_classifier, batch_size, device):
    '''
    Uses the transformer encoder to encode sentences.
    Arguments:
        ds_list: list of instances of singlesc_models.Single_SC_Dataset.
        sentence_classifier: instance of HierarchicalSC.
    Returns:
        List of tensors representing the sentence embeddings.
        List of tensort representing the sentence targets.
    '''
    targets = []    # a tensor by doc
    embeddings = [] # a tensor by doc
    for ds in ds_list: # iterates datasets/documents
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        targets_doc = []
        embeddings_doc = []
        for batch in dl: # iterates batches of sentences in current document
            # first-level encoding
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets_doc.append(batch['target'].to(device))
            embeddings_doc.append(
                sentence_classifier.encode_batch(ids, mask).to(device)
            )
            # end batch
        targets.append(torch.hstack(targets_doc))
        embeddings.append(torch.vstack(embeddings_doc))
        # end doc
    return embeddings, targets

def evaluate(model, rnn_ds, loss_function, batch_size, device):
    """
    Evaluates a provided model.
    Arguments:
        model: the model to be evaluated. An instance of HierarchicalSC.
        rnn_ds: instance of TensorList_Dataset.
        loss_function: instance of the loss function used to train the model.
        batch_size: batch size.
        device: device where the model is located.
    Returns:
        eval_loss (float): the computed test loss score.
        precision (float): the computed test Precision score.
        recall (float): the computed test Recall score.
        f1 (float): the computed test F1 score.
        confusion_matrix: the computed test confusion matrix.
    """
    predictions = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    eval_loss = []
    model.eval()
    with torch.no_grad():
        
        dl = torch.utils.data.DataLoader(rnn_ds, batch_size=batch_size, collate_fn=collate_batch_TensorList)
        for embeddings_batch, y_true_batch in dl: # iterates batches of documents
            # second-level encoding and classification
            y_hat_batch = model(embeddings_batch)
            # ignores classes with negative ID
            y_true_batch = torch.hstack(y_true_batch)
            idx_valid = (y_true_batch >= 0).nonzero().squeeze()
            y_true_batch_valid = y_true_batch[idx_valid]
            y_hat_batch_valid = y_hat_batch[idx_valid]
            # getting loss and valid predictions
            loss = loss_function(y_hat_batch_valid, y_true_batch_valid)
            eval_loss.append(loss.item())
            predictions_batch_valid = y_hat_batch_valid.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_batch_valid))
            y_true = torch.cat((y_true, y_true_batch_valid))
            # end batch
        
        '''
        for ds in test_ds_lst: # iterates datasets/documents
            dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            y_true_doc = []
            embeddings_doc = []
            for batch in dl: # iterates batches of sentences in current document
                # first-level encoding
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                y_true_doc.append(batch['target'].to(device))
                embeddings_doc.append(
                    model.encode_batch(ids, mask)
                )
                # end batch
            y_true_doc = torch.hstack(y_true_doc)
            embeddings_doc = torch.vstack(embeddings_doc)
            # second-level encoding and classification
            y_hat_doc = model(embeddings_doc)
            # ignores classes with negative ID
            idx_valid = (y_true_doc >= 0).nonzero().squeeze()
            y_true_doc_valid = y_true_doc[idx_valid]
            y_hat_doc_valid = y_hat_doc[idx_valid]
            # getting loss and valid predictions
            loss = loss_function(y_hat_doc_valid, y_true_doc_valid)
            eval_loss.append(loss.item())
            predictions_doc_valid = y_hat_doc_valid.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_doc_valid))
            y_true = torch.cat((y_true, y_true_doc_valid))
        '''
        predictions = predictions.detach().to('cpu').numpy()
        y_true = y_true.detach().to('cpu').numpy()
    eval_loss = np.array(eval_loss).mean()
    t_metrics_macro = precision_recall_fscore_support(
        y_true, 
        predictions, 
        average='macro', 
        zero_division=0
    )
    cm = confusion_matrix(
        y_true, 
        predictions
    )
    
    return eval_loss, t_metrics_macro[0], t_metrics_macro[1], t_metrics_macro[2], cm

def fit(train_params, ds_train_lst, ds_test_lst, transformer_encoder, device):
    """
    Creates and train an instance of HierarchicalSC.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train_lst: list of instances of singlesc_models.Single_SC_Dataset storing the train data.
        ds_test_lst: list of instances of singlesc_models.Single_SC_Dataset storing the test data.
        transformer_encoder: an instance of singlesc_models.SingleSentenceEncoder_BERT.
        device: device where the model should be located.
    """
    learning_rate_classifier = train_params['learning_rate_classifier']
    weight_decay = train_params['weight_decay']
    n_epochs_classifier = train_params['n_epochs_classifier']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    lstm_hidden_dim = train_params['lstm_hidden_dim']
    batch_size = train_params['batch_size']
    
    # creating model
    sentence_classifier = HierarchicalSC(transformer_encoder, n_classes, dropout_rate, embedding_dim, lstm_hidden_dim).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        sentence_classifier.parameters(), 
        lr=learning_rate_classifier, 
        betas=(0.9, 0.999), 
        eps=train_params['eps'], 
        weight_decay=weight_decay
    )
    # computing n_training_steps
    n_training_steps = 0
    for ds in ds_train_lst:
        n_training_steps += math.ceil(len(ds) / batch_size)
    n_training_steps = n_training_steps * n_epochs_classifier
    
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = n_training_steps
    )
    
    metrics = {} # key: epoch number, value: numpy tensor storing train loss, test loss, P macro, R macro, F1 macro
    confusion_matrices = {} # key: epoch number, value: scikit-learn's confusion matrix
    start_train = time.perf_counter()
    
    # we get sentence embeddings and targets outside epoch loop to avoid input into transformer encoder
    # and speed up the training
    print('  Encoding sentences...', end='')
    embeddings_train, y_true_train = first_encoding(ds_train_lst, sentence_classifier, batch_size, device)
    embeddings_test, y_true_test = first_encoding(ds_test_lst, sentence_classifier, batch_size, device)
    print(' Done!')
    
    rnn_ds_train = TensorList_Dataset(embeddings_train, y_true_train)
    rnn_ds_test = TensorList_Dataset(embeddings_test, y_true_test)
    
    for epoch in range(1, n_epochs_classifier + 1):
        print(f'  Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = []
        sentence_classifier.train()
        
        dl = torch.utils.data.DataLoader(rnn_ds_train, batch_size=batch_size, collate_fn=collate_batch_TensorList)
        for embeddings_batch, y_true_batch in dl: # iterates batches of documents
            '''
            print('  len(embeddings_batch):', len(embeddings_batch))
            print('  embeddings_batch[0].shape:', embeddings_batch[0].shape)
            print('  embeddings_batch[1].shape:', embeddings_batch[1].shape)
            print('  len(y_true_batch):', len(y_true_batch))
            print('  y_true_batch[0].shape):', y_true_batch[0].shape)
            print('  y_true_batch[1].shape):', y_true_batch[1].shape)
            '''
            # second-level encoding and classification
            y_hat_batch = sentence_classifier(embeddings_batch)
            # ignores classes with negative ID
            y_true_batch = torch.hstack(y_true_batch)
            idx_valid = (y_true_batch >= 0).nonzero().squeeze()
            y_true_batch_valid = y_true_batch[idx_valid]
            y_hat_batch_valid = y_hat_batch[idx_valid]
            loss = criterion(y_hat_batch_valid, y_true_batch_valid)
            epoch_loss.append(loss.item())
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sentence_classifier.parameters(), 1.0)
            # updating weights and learning rate
            optimizer.step()
            lr_scheduler.step()
            # end batch
            
        epoch_loss = np.array(epoch_loss).mean()
        # evaluation
        optimizer.zero_grad()
        eval_loss, p_macro, r_macro, f1_macro, cm = evaluate(
            sentence_classifier, 
            rnn_ds_test, 
            criterion, 
            batch_size, 
            device
        )
        #storing metrics
        metrics[epoch] = np.array([epoch_loss, eval_loss, p_macro, r_macro, f1_macro])
        confusion_matrices[epoch] = cm
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
        # end epoch
    end_train = time.perf_counter()
    
    return metrics, confusion_matrices, time.strftime("%Hh%Mm%Ss", time.gmtime(end_train - start_train))
