from os import listdir
import pandas as pd
import numpy as np
import random, torch, singlesc_models, hierarchical_models, transformers, time, data_manager
from datetime import datetime

def get_datasets(train_params, dic_docs, labels_to_idx, tokenizer):
    '''
    Returns a list of datasets where which one represents a single document.
    '''
    local_dic_docs = dic_docs
    if train_params.get('n_documents') is not None: # used in tests to speed up the train procedure
        n_documents = train_params.get('n_documents')
        local_dic_docs = {k: local_dic_docs[k] for k in sorted(local_dic_docs.keys())[:n_documents]}
    
    lst_ds = []
    for df in local_dic_docs.values():
        lst_ds.append(
            singlesc_models.Single_SC_Dataset(df['sentence'].to_list(), df['label'].to_list(), labels_to_idx, tokenizer)
        )
    return lst_ds

def fine_tune(train_params, ds_encoder_train, device):
    '''
    Fine-tunes a transformer model.
    '''
    encoder_id = train_params['encoder_id']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    n_epochs_encoder = train_params['n_epochs_encoder']
    stop_epoch_encoder = train_params['stop_epoch_encoder']
    batch_size = train_params['batch_size']
    learning_rate_encoder = train_params['learning_rate_encoder']
    weight_decay = train_params['weight_decay']
    
    assert stop_epoch_encoder <= n_epochs_encoder
    
    encoder = singlesc_models.SingleSC_BERT(
        encoder_id, 
        n_classes, 
        dropout_rate, 
        embedding_dim
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        encoder.parameters(), 
        lr=learning_rate_encoder, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=weight_decay
    )
    dl_train_encoder = torch.utils.data.DataLoader(ds_encoder_train, batch_size=batch_size, shuffle=True)
    num_training_steps = len(dl_train_encoder) * n_epochs_encoder
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = num_training_steps
    )

    print('=> Start of encoder training')
    for epoch in range(1, stop_epoch_encoder + 1):
        print(f'  Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        encoder.train()
        for train_data in dl_train_encoder:
            optimizer.zero_grad()
            ids = train_data['ids'].to(device)
            mask = train_data['mask'].to(device)
            y_hat = encoder(ids, mask)
            y_true = train_data['target'].to(device)
            loss = criterion(y_hat, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
    return encoder.encoder

def evaluate_BERT(train_params):
    # time tag
    model_reference = train_params['model_reference']
    time_tag = f'{model_reference}_{datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")}'
    train_params['time_tag'] = time_tag
    
    # loading dataset
    dataset_name = train_params['dataset']
    data_loader = data_manager.get_data_manager(dataset_name)
    
    # setting labels
    labels_to_idx = data_loader.get_labels_to_idx()
    labels = data_loader.get_valid_labels(labels_to_idx)
    train_params['n_classes'] = len(labels)
    
    # tokenizer
    encoder_id = train_params['encoder_id']
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_id)
    
    # loading data
    dic_docs_train, dic_docs_dev, dic_docs_test = data_loader.get_data()
    
    # dataset object to train the transformer encoder
    if train_params.get('n_documents') is not None: # used in tests to speed up the train procedure
        n_documents = train_params.get('n_documents')
        temp_docs_train = {k: dic_docs_train[k] for k in sorted(dic_docs_train.keys())[:n_documents]}
        ds_encoder_train = singlesc_models.get_dataset(temp_docs_train, labels_to_idx, tokenizer)
    else:
        ds_encoder_train = singlesc_models.get_dataset(dic_docs_train, labels_to_idx, tokenizer)
    
    # dataset objects to train and evalutate the classifier
    use_dev_set = train_params['use_dev_set']
    if use_dev_set:
        dic_docs_test = dic_docs_dev
    lst_ds_train = get_datasets(train_params, dic_docs_train, labels_to_idx, tokenizer)
    lst_ds_test = get_datasets(train_params, dic_docs_test, labels_to_idx, tokenizer)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    raw_metrics = {} # key: epoch, value: numpy tensor of shape (n_iterations, 5)
    confusion_matrices = {} # key: iteration_id, value: dictionary (key: epoch, value: confusion matrix)
    start_total = time.perf_counter()
    seeds = [(42 + i * 10) for i in range(train_params['n_iterations'])]
    
    for i, seed_val in enumerate(seeds):
        print(f'==== Started iteration {i + 1} ====')
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        
        # first-level sentence encoder
        use_mock = train_params['use_mock']
        embedding_dim = train_params['embedding_dim']
        if use_mock:
            transformer_encoder = hierarchical_models.MockEncoder(embedding_dim).to(device)
        else:
            transformer_encoder = fine_tune(train_params, ds_encoder_train, device)
        
        # classifier training
        print('=> Start of classifier training')
        iteration_metrics, cm, train_time = hierarchical_models.fit(train_params, lst_ds_train, lst_ds_test, transformer_encoder, device)
        confusion_matrices[i] = cm
        for epoch, scores in iteration_metrics.items():
            epoch_metrics = raw_metrics.get(epoch, None)
            if epoch_metrics is None:
                raw_metrics[epoch] = scores.reshape(1,-1)
            else:
                raw_metrics[epoch] = np.vstack((epoch_metrics, scores))
        print('  Iteration time: ', train_time)

    metrics = pd.DataFrame(columns=[
        'Epoch', 'Train loss', 'std', 'Test loss', 'std', 
        'P (macro)', 'P std', 'R (macro)', 'R std', 'F1 (macro)', 'F1 std'
    ])
    for i, (epoch, scores) in enumerate(raw_metrics.items()):
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        metrics.loc[i] = [
            f'{epoch}', 
            f'{mean[0]:.6f}', f'{std[0]:.6f}',    # train loss
            f'{mean[1]:.6f}', f'{std[1]:.6f}',    # test loss
            f'{mean[2]:.4f}', f'{std[2]:.4f}',    # precision (macro)
            f'{mean[3]:.4f}', f'{std[3]:.4f}',    # recall (macro)
            f'{mean[4]:.4f}', f'{std[4]:.4f}'     # f1 (macro)
        ]
    
    end_total = time.perf_counter()
    total_time = time.strftime("%Hh%Mm%Ss", time.gmtime(end_total - start_total))
    print('End of evaluation. Total time:', total_time)
    save_report(
        metrics, raw_metrics, labels, confusion_matrices, 
        f'{"development set" if use_dev_set else "test set"} ({len(seeds)} random seeds)', 
        train_params, total_time, device, time_tag
    )

def save_report(
    avg_metrics, complete_metrics, labels, 
    confusion_matrices, evaluation, train_params, train_time, device, time_tag):
    """
    Arguments:
        avg_metrics : A pandas Dataframe with the averaged metrics.
        complete_metrics : A dictionary with the metrics by epoch. The key indicates the epoch. 
                            Each value must be a numpy tensor of shape (n_iterations, 5).
        labels : list of all labels.
        confusion_matrices : A dictionary => key: iteration_id, value: dictionary (key: epoch, value: confusion matrix)
        evaluation : the kind of evalutaion (string). Cross-validation or Holdout.
        train_params : A dictionary.
        train_time : total time spent on training/evaluation (string).
        device : ID of GPU device
        time_tag : time tag to be appended to report file name.
    """
    model_reference = train_params['model_reference']
    dataset_name = train_params["dataset"]
    report = (
        'RESULTS REPORT (SINGLE SENTENCE CLASSIFICATION)\n'
        f'Model: {model_reference}\n'
        f'Encoder: {train_params["encoder_id"] if not train_params["use_mock"] else "MOCK MODEL"}\n'
        f'Dataset: {dataset_name}\n'
        f'Evaluation: {evaluation}\n'
        f'Max sequence length: {train_params["max_seq_len"]}\n'
        f'Batch size: {train_params["batch_size"]}\n'
        f'Dropout rate: {train_params["dropout_rate"]}\n'
        f'Learning rate (encoder): {train_params["learning_rate_encoder"]}\n'
        f'Number of epochs (encoder): {train_params["n_epochs_encoder"]}\n'
        f'Stop epoch (encoder): {train_params["stop_epoch_encoder"]}\n'
        f'Embedding dim (encoder): {train_params["embedding_dim"]}\n'
        f'Learning rate (classifier): {train_params["learning_rate_classifier"]}\n'
        f'Number of epochs (classifier): {train_params["n_epochs_classifier"]}\n'
        f'Hidden dim (classifier): {train_params["lstm_hidden_dim"]}\n'
        f'Adam Epsilon: {train_params["eps"]}\n'
        f'Weight decay: {train_params["weight_decay"]}\n'
        f'Train time: {train_time}\n'
    )
    
    if torch.cuda.is_available():
        report += f'GPU name: {torch.cuda.get_device_name(device)}\n'
        memory_in_bytes = torch.cuda.get_device_properties(device).total_memory
        memory_in_gb = round((memory_in_bytes/1024)/1024/1024,2)
        report += f'GPU memory: {memory_in_gb}\n'
    
    report += '\nAverages:\n'
    report += avg_metrics.to_string(index=False, justify='center')
    
    report += '\n\n*** Detailed report ***\n'
    
    report += f'\nConfusion matrices\n{"-"*18}\n'
    for i, label in enumerate(labels):
        report += f'{label}: {i} \n'
    for iteration_id, cm_dic in confusion_matrices.items():
        report += f'=> Iteration {iteration_id}:\n'
        # reports only the confusion matrix of the last epoch
        n_epochs_classifier = train_params["n_epochs_classifier"]
        report += f'Epoch {n_epochs_classifier}:\n{cm_dic[n_epochs_classifier]}\n'

    report += f'\nScores\n{"-"*6}\n'
    for epoch, scores in complete_metrics.items():
        df = pd.DataFrame(
            scores, 
            columns=['Train loss', 'Test loss', 'P (macro)', 'R (macro)', 'F1 (macro)'], 
            index=[f'Iteration {i}' for i in range(scores.shape[0])])
        report += f'Epoch: {epoch}\n' + df.to_string() + '\n\n'
    
    with open(f'./reports/{dataset_name}/rep-hierarchical-{time_tag}.txt', 'w') as f:
        f.write(report)
