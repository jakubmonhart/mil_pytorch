import numpy
from torch.utils.data import DataLoader
from sklearn import metrics
import torch

def accuracy(pred, target, threshold = 0):
	'''
	'''

	pred = pred.detach().numpy()
	target = target.detach().numpy()

	pred[pred >= threshold] = 1
	pred[pred < threshold] = -1

	return numpy.sum(target == pred)/target.shape[0]



def eer(pred, labels):
	''' 
	'''

	fpr, tpr, threshold = metrics.roc_curve(labels.detach(), pred.detach(), pos_label=1)
	fnr = 1 - tpr
	EER_fpr = fpr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
	EER_fnr = fnr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
	return EER_fpr, EER_fnr


def evaluation(model, criterion, train_dl, test_dl, device = 'cpu'):
    print('EVALUATION:')
    # Train set accuracy
    train_data = torch.empty(size = (0,), dtype = torch.double).to(device)
    train_ids = torch.empty(size = (0,), dtype = torch.long).to(device)
    train_labels = torch.empty(size = (0,), dtype = torch.long).to(device)

    for data, ids, labels in train_dl:
        train_data = torch.cat((train_data, data))
        train_ids = torch.cat((train_ids, ids))
        train_labels = torch.cat((train_labels, labels))

    pred = model((train_data, train_ids)) 
    loss = criterion(pred[:,0], train_labels)

    pred = pred.cpu()
    train_labels = train_labels.cpu()    
    eer_fpr, eer_fnr = eer(pred[:,0], train_labels)
    print('TRAIN')
    print(' - loss_train: {}'.format(loss.item()))
    print(' - acc_train: {}'.format(accuracy(pred[:,0], train_labels)))
    print(' - eer_fpr_train: {}'.format(eer_fpr))
    print(' - eer_fnr_train: {}'.format(eer_fnr))
        
    # Test set accuracy
    for data, ids, labels in test_dl:
        pred = model((data, ids))
        loss = criterion(pred[:,0], labels)
        pred = pred.cpu()
        labels = labels.cpu()
        eer_fpr, eer_fnr = eer(pred[:,0], labels)
    
    print('TEST')
    print(' - loss_test: {}'.format(loss.item()))
    print(' - acc_test: {}'.format(accuracy(pred[:,0], labels)))
    print(' - eer_fpr_test: {}'.format(eer_fpr))
    print(' - eer_fnr_test: {}'.format(eer_fnr))
