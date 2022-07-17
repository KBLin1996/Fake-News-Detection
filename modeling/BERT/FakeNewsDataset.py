import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


# Load our dataset with BertTokenizer and separate into training and testing set
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, data_type='val'):
        # This will download the pretrained BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if data_type == 'train':
            # Split train and validation dataset and get train
            # train_test_split() => return (train_x, test_x, train_y, test_y)
            self.text, _, self.label, _ = train_test_split(texts, labels, test_size=0.2, random_state=42)
        elif data_type:
            # Split train and validation dataset and get validate
            _, self.text, _, self.label = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Pre-tokenizing all sentences
        print('Tokenizing...', end = '')
        self.tokenized_plots = list()
        for i in range(0, len(self.text)):
            text = self.text[i]

            # Method to generally tokenize it in projects:
            #
            # [Parameters]
            #
            # truncation => as BERT can only accept/take as input only 512 tokens at a time, we must specify
            #               the truncation parameter to True
            # add_special_tokens => is just for BERT to add tokens like the start, end, [SEP], and [CLS] tokens
            # return_tensors=“pt” => is just for the tokenizer to return PyTorch tensors. If you don’t want this
            #                        to happen(maybe you want it to return a list), then you can remove the
            #                        parameter and it will return lists
            # return_attention_mask => The attention mask is a binary tensor indicating the position of the
            #                          padded indices so that the model does not attend to them.
            encoded_text = self.tokenizer.encode_plus(
                text, add_special_tokens = True, truncation = True, 
                max_length = 512, padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt')

            self.tokenized_plots.append(encoded_text)
        print(' finished')
            
    def __getitem__(self, index: int):
        # text => tensor([[ ... ]]) contains text ids in the tokenizers
        text = self.tokenized_plots[index]['input_ids'][0]
        text_mask = self.tokenized_plots[index]['attention_mask'][0]

        # Encode labels in a binary vector.
        #label_vector = torch.zeros(2)
        # Expand the dimension of label
        # => initially: label = 0 or 1
        # => now: label_vector = [1, 0] if label == 0, [0, 1] if label == 1
        #label_vector[self.label[index]] = 1
        label_vector = self.label[index]

        return text, text_mask, label_vector

    def get_data(self, index: int):
        text, label = self.data[index]
        return text, label

    def __len__(self):
        return len(self.text)