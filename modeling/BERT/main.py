import test
import utils
import train
import validate
import FakeNewsDataset
import data_preprocessing


if __name__ == "__main__":
    # Data preprocessing
    texts, labels = data_preprocessing.ReadCSV(include_date=False)

    # Fetch training and testing data
    train_data = FakeNewsDataset(texts, labels, data_type='train')
    val_data = FakeNewsDataset(texts, labels, data_type='val')

    # Training process
    train_loader, val_loader = utils.loaders()
    train(train_loader, val_loader)

    # Validation process
    predictions = list()
    actual_labels = list()
    predictions, actual_labels, incorrect_samples = validate()

    utils.print_val_results(actual_labels, predictions)
    utils.print_incorrect_val_data(val_data, incorrect_samples)

    # Customized testing process
    input_title = "The news team simulates deaths from COVID-19 in order to say there are more people dying than really is"
    testing_model = utils.load_pretrained_model()
    test.testing(testing_model, input_title)