import utils
import torch
import data_preprocessing
import FakeNewsDataset


# See if the data was inaccurately predicted
def validate_data(predicted_val, label, val_num):
    # Why sigmoid instead of softmax?
    # => The sigmoid function is used for the two-class logistic regression
    # => The softmax function is used for the multiclass logistic regression
    
    val = predicted_val.sigmoid()
    pred_label = val > 0.5 # Anything with sigmoid > 0.5 is 1.

    true_predicted = True
    if pred_label != label:
        true_predicted = False
    #     print(val_num)
    #     print(pred_label.item())
    #     print(bool(label.item()))

    return pred_label.item(), bool(label.item()), true_predicted


# Find out all incorrect data
def find_incorrect_data(val_loader, model, batch_size=10):
    predictions = list()
    actual_labels = list()
    incorrect_samples = list()

    with torch.no_grad():
        for (batch_id, (texts, text_masks, labels)) in enumerate(val_loader):
            # Move to GPU.
            texts = texts.cuda()
            text_masks = text_masks.cuda()
            labels = labels[:, None]
            labels = labels.cuda()

            # Compute predictions.
            predicted = model(texts, text_masks)
            
            for i in range(len(predicted.logits.data)):
                val_num = batch_id * batch_size + i
                prediction, label, true_predicted = validate_data(predicted.logits.data[i], labels[i], val_num)

                predictions.append(prediction)
                actual_labels.append(label)

                if not true_predicted:
                    incorrect_samples.append(val_num)
                
    return predictions, actual_labels, incorrect_samples


# Main validation code
def validate():
    batch_size = 10
    raw_texts, raw_labels = data_preprocessing.ReadCSV(include_date=False)

    val_data = FakeNewsDataset(raw_texts, raw_labels, data_type='val')
    print(f"Validation size: {len(val_data)} samples\n")

    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size = batch_size,
                                            shuffle = False)

    model = utils.load_pretrained_model()
    return find_incorrect_data(val_loader, model, batch_size=10)