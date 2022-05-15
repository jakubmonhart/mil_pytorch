

from mil.utils import get_raw_train_test_data
from mil.dataloader import get_train_test_dataloader
from mil.nn.models import MI_Net, mi_Net, MI_Net_RC, MI_Net_Attention
from mil.train import train, eval_model
from mil.hyper_parameters import learning_rates, weight_decays


if __name__ == "__main__":
    batch_size = 1
    data_name  = "Musk1"
    data_dir = f'{data_name}.xlsx'
    test_index_dir = f"{data_name}.csv_rep1_fold1.txt"

    train_data, test_data = get_raw_train_test_data(data_dir=data_dir, test_index_dir=test_index_dir)
    train_dl, test_dl = get_train_test_dataloader(train_data=train_data, test_data=test_data, train_batch_size=batch_size, 
        test_batch_size=batch_size)

    model = MI_Net()  # build the model
    model = train(model=model, train_dl=train_dl, learning_rate=learning_rates[data_name], weight_decay=weight_decays[data_name])
    train_acc, train_pred_prob = eval_model(model=model, )
    
