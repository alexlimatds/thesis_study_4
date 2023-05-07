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
        return cls_embeddings # cls_embeddings.shape: (batch_size, embedd_dim)
    
    def forward(self, sent_embeddings):
        '''
        Each call to this method process one document represented as a sequence of sentence embeddings 
        yelded by the encode_batch method.
        The sentence embeddings are fed to a BiLSTM network to get embeddings enriched with document context.
        This method returns one logit tensor for each sentence in the document.
        Arguments:
            sent_embeddings : tensor of shape (n_sentences, embedding_dim)
        Returns:
            logits : tensor of shape (n_sentences, n_classes)
        '''
        sent_embeddings = self.dropout(sent_embeddings)
        lstm_embeddings, _ = self.bilstm(
            sent_embeddings.unsqueeze(0)  # LSTM requires batch dimension so we add it with unsqueeze
        )
        # squeeze to remove batch dimension
        lstm_embeddings = lstm_embeddings.squeeze(0) # lstm_embeddings.shape: (n_sentences, lstm_hidden_dim)
        
        # classification
        logits = self.classifier(lstm_embeddings)   # logits.shape: (n_sentences, n_classes)

        return logits

def evaluate(model, test_ds_lst, loss_function, batch_size, device):
    """
    Evaluates a provided model.
    Arguments:
        model: the model to be evaluated.
        test_ds_lst: list of instances of DFCSC_Dataset storing the test data.
        loss_function: instance of the loss function used to train the model.
        batch_size: 
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
    sentence_classifier = HierarchicalSC(transformer_encoder, n_classes, dropout_rate, embedding_dim, lstm_hidden_dim)
    
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
    for epoch in range(1, n_epochs_classifier + 1):
        print(f'  Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = []
        sentence_classifier.train()
        for ds in ds_train_lst: # iterates datasets/documents
            optimizer.zero_grad()
            dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            y_true_doc = []
            embeddings_doc = []
            for batch in dl: # iterates batches of sentences in current document
                # first-level encoding
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                y_true_doc.append(batch['target'].to(device))
                embeddings_doc.append(
                    sentence_classifier.encode_batch(ids, mask).to(device)
                )
                # end batch
            y_true_doc = torch.hstack(y_true_doc)
            embeddings_doc = torch.vstack(embeddings_doc)
            # second-level encoding and classification
            y_hat_doc = sentence_classifier(embeddings_doc)
            # ignores classes with negative ID
            idx_valid = (y_true_doc >= 0).nonzero().squeeze()
            y_true_doc_valid = y_true_doc[idx_valid]
            y_hat_doc_valid = y_hat_doc[idx_valid]
            loss = criterion(y_hat_doc_valid, y_true_doc_valid)
            epoch_loss.append(loss.item())
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sentence_classifier.parameters(), 1.0)
            # updating weights and learning rate
            optimizer.step()
            lr_scheduler.step()
            # end document processing
        epoch_loss = np.array(epoch_loss).mean()
        # evaluation
        optimizer.zero_grad()
        eval_loss, p_macro, r_macro, f1_macro, cm = evaluate(
            sentence_classifier, 
            ds_test_lst, 
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
