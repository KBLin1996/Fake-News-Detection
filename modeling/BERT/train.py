import utils
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup


# Calculating the batch accuracy
def batch_accuracy(predicted, labels):
    # Why sigmoid instead of softmax?
    # => The sigmoid function is used for the two-class logistic regression
    # => The softmax function is used for the multiclass logistic regression
    
    vals = predicted.data.sigmoid()
    pred_vals = vals > 0.5 # Anything with sigmoid > 0.5 is 1.

    #print(f"Prediction Value: {pred_vals}")
    #print(f"Labels: {labels}")

    return (pred_vals == labels).sum(dim = 1)


# Training process
def train(train_loader, val_loader):
    # Load model from utils
    model = utils.load_model()

    # Training Loop.
    num_epochs = 2

    # Following recommendations from the BERT paper and also this
    # blog post https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-6, eps = 1e-8)

    # Total number of training steps is [number of batches] * [number of epochs]. 
    # (Note that this is not the same as the number of training samples)
    total_steps = len(train_loader) * num_epochs

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    # Binary Cross-Entroy Loss (BCE Loss).
    # Please see documentation here:
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # This loss combines a Sigmoid layer and the BCELoss in one single class
    cost_function = nn.BCEWithLogitsLoss(reduction = 'none')

    best_accuracy = 0
    current_step = 0
    for epoch in range(0, num_epochs):
        cumulative_accuracy = 0
        cumulative_loss = 0
        num_samples = 0
        logs = {}
        model.train()
        for (batch_id, (texts, text_masks, labels)) in enumerate(train_loader):
            # Move to GPU.
            texts = texts.cuda()
            text_masks = text_masks.cuda()
            labels = labels[:, None]
            labels = labels.cuda()

            # Compute predictions.
            # Returns:
            # loss => Language modeling loss (for next-token prediction)
            # logits => Classification (or regression if config.num_labels==1) scores (before SoftMax)
            predicted = model(texts, text_masks)

            # Compute loss.
            loss = cost_function(predicted.logits, labels.float())

            # Compute cumulative loss and accuracy.
            # Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
            cumulative_loss += loss.data.sum().item()

            cumulative_accuracy += batch_accuracy(predicted.logits, labels).sum().item()
            num_samples += texts.size(0)

            # Backpropagation and SGD update step.
            model.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
            if batch_id % 100 == 0:
                logs['loss'] = cumulative_loss / num_samples
                logs['accuracy'] = cumulative_accuracy / num_samples
                current_step += 1
        
        cumulative_accuracy = 0
        cumulative_loss = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for (batch_id, (texts, text_masks, labels)) in enumerate(val_loader):
                # Move to GPU.
                texts = texts.cuda()
                text_masks = text_masks.cuda()
                labels = labels[:, None]
                labels = labels.cuda()

                # Compute predictions.
                predicted = model(texts, text_masks)

                # Compute loss.
                loss = cost_function(predicted.logits, labels.float())

                # Compute cumulative loss and top-5 accuracy.
                cumulative_loss += loss.data.sum().item()
                cumulative_accuracy += batch_accuracy(predicted.logits, labels).sum().item()
                num_samples += texts.size(0)

                if (1 + batch_id) % 100 == 0:
                    logs['val_loss'] = cumulative_loss / num_samples
                    logs['val_accuracy'] = cumulative_accuracy / num_samples

        # Advance scheduler.
        if scheduler != -1:
            scheduler.step()

        # Save the parameters for the best accuracy on the validation set so far.
        if logs['val_accuracy'] > best_accuracy:
            best_accuracy = logs['val_accuracy']
            torch.save(model.state_dict(), 'best_model_so_far.pth') 