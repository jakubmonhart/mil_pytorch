

from mil.utils import get_raw_train_test_data
from mil.dataloader import get_train_test_dataloader
from mil.nn.models import MI_NET, mi_NET, MI_NET_RC, MI_NET_Attention
from mil.train import train, eval_model



if __name__ == "__main__":
    batch_size = 1
    data_dir = 'Musk1.xlsx'
    test_index_dir = "Musk1.csv_rep1_fold1.txt"

    train_data, test_data = get_raw_train_test_data(data_dir=data_dir, test_index_dir=test_index_dir)
    train_dl, test_dl = get_train_test_dataloader(train_data=train_data, test_data=test_data, train_batch_size=batch_size, 
        test_batch_size=batch_size)

    model = MI_NET()  # build the model
    model = train(model=model, train_dl=train_dl)
    train_acc, train_pred_prob = eval_model(model=model, )
    
