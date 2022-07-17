from transformers import BertTokenizer


def test_tokenizer(title):
    # This will download the pretrained BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenizing the input sentence
    print('Tokenizing the test statement...', end = '')
    tokenized_plots = list()

    # Method to generally tokenize it in projects:
    #
    # [Parameters]
    #
    # truncation => as BERT can only accept/take as input only 512 tokens at a time, we must specify the truncation parameter to True
    # add_special_tokens => is just for BERT to add tokens like the start, end, [SEP], and [CLS] tokens
    # return_tensors=“pt” => is just for the tokenizer to return PyTorch tensors. If you don’t want this to happen(maybe you want it to return a list), then you can remove the parameter and it will return lists
    # return_attention_mask => The attention mask is a binary tensor indicating the position of the padded indices so that the model does not attend to them.
    encoded_text = tokenizer.encode_plus(
                    title,
                    add_special_tokens = True,
                    truncation = True,
                    max_length = 512,
                    padding = 'max_length',
                    return_attention_mask = True,
                    return_tensors = 'pt'
                )
    print(' finished\n')

    text = encoded_text['input_ids'][0]
    text_mask = encoded_text['attention_mask'][0]

    return text, text_mask


def testing(model, input_title):
    text, text_mask = test_tokenizer(input_title)

    # Move to GPU.
    text = text.cuda()
    text_mask = text_mask.cuda()

    # Compute the prediction
    predicted_val = model(text.unsqueeze(0), text_mask.unsqueeze(0))

    # Anything with sigmoid > 0.5 is 1
    realiable_val = predicted_val.logits.data.sigmoid().item()

    prediction = (realiable_val > 0.5)
    print(f"Input Title: {input_title}")

    if prediction == True:
        print(f"Prediction: {prediction} News")
    elif prediction == False:
        print(f"Prediction: Fake News")


    # Show our prediction confidence
    if realiable_val <= 0.5:
        print("Confidence: {:.2f}%".format((0.5 - realiable_val) * 100 / 0.5))
    else:
        print("Confidence: {:.2f}%".format(-(0.5 - realiable_val) * 100 / 0.5))