import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


# Create pytorch train loader and validation loader for the training and the validation process
def loaders(train_data, val_data):
    # Batch size is a liming factor on constrained resources.
    # only GPUs with a large memory can hold large batches.
    batch_size = 10


    # If you load your samples in the Dataset on CPU and would like to push it during training to
    # the GPU, you can speed up the host to device transfer by enabling pin_memory.

    # This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.
    train_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size = batch_size,
                                            shuffle = True,
                                            pin_memory = True,
                                            num_workers = 2)

    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size = batch_size,
                                            shuffle = False)

    return train_loader, val_loader


# Loading BERT pre-trained classifier
def load_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                        num_labels = 1, output_attentions = False, 
                        output_hidden_states = False)

    return model


# Load the model we've trained
def load_pretrained_model():
    model = load_model()

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Load our pretrained model
    model.load_state_dict(torch.load('best_model_so_far.pth'))
    model.eval()


# Print validation results
def print_val_results(actual_labels, predictions):
    # Specify label "True" is the first element
    score = precision_recall_fscore_support(actual_labels, predictions, labels=[True, False], average='binary')

    print(f"Confusion Matrix:\n{confusion_matrix(actual_labels, predictions, labels=[True, False])}\n")
    print(f"Precision: {score[0]}")
    print(f"Recall: {score[1]}")
    print(f"F1-Score: {f1_score(actual_labels, predictions, labels=[True, False])}")


def print_incorrect_val_data(val_data, incorrect_samples):
    val_texts = val_data.text
    val_labels = val_data.label

    for val_num in incorrect_samples:
        #print(val_num)
        print(f"Text: {val_texts[val_num]}")
        print(f"Prediction: {not bool(val_labels[val_num])}")
        print(f"Actual Label: {bool(val_labels[val_num])}\n\n")