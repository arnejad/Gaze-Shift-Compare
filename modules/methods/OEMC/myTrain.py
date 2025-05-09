# This script is a copy of main.py from the original repository of OEMC with minor changes from us ...
# ...to make adaptable to our framework

import numpy as np
from tcn import TCN
from cnn_blstm import CNN_BLSTM
from cnn_lstm import CNN_LSTM
import torch
import torch.nn.functional as F
import preprocessor
import os
import numpy as np
import random
import scorer
import argparse
import copy

from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.dataloader import listRecNames
from modules.methods.ACEDNV.modules.scorer import score

from config import INP_DIR, LABELER, ET_SAMPLING_RATE, OEMC_MODEL


def train(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        loss = F.nll_loss(output, y, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        preds = pred.view(pred.numel())
        return preds, y, loss


def f1_score(preds, labels, class_id):
    '''
    preds: precictions made by the network
    labels: list of expected targets
    class_id: corresponding id of the class
    '''
    true_count = torch.eq(labels, class_id).sum()
    true_positive = torch.logical_and(torch.eq(labels, preds),
                                      torch.eq(labels, class_id)).sum().float()
    precision = torch.div(true_positive, torch.eq(preds, class_id).sum().float())
    precision = torch.where(torch.isnan(precision),
                            torch.zeros_like(precision).type_as(true_positive),
                            precision)
    recall = torch.div(true_positive, true_count)
    f1 = 2*precision*recall / (precision+recall)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive),f1)
    return f1.item()


def save_test_output(model_path, preds, labels):
    output_path = 'outputs/' + model_path
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    preds = preds.cpu()
    labels = labels.cpu()
    np.savez(output_path, pred=preds.numpy(), gt=labels.numpy())


def predict(model, num_test_batches, batch_size, X_val, Y_val, timesteps, pproc):
    total_pred = torch.Tensor([]).cuda()
    total_label = torch.Tensor([]).cuda()
    test_loss = 0
    test_size = len(Y_val)
    for k in range(num_test_batches):
        start, end = k*batch_size, (k+1)*batch_size
        if start == 0:
            start = timesteps
        X,Y = pproc.create_batches(X_val, Y_val, start, end, timesteps)
        preds, labels, loss = test(model, X, Y)
        test_loss += loss
        total_pred = torch.cat([total_pred, preds], dim=0)
        total_label = torch.cat([total_label, labels], dim=0)
    test_loss /= test_size
    return test_loss, total_pred, total_label


def print_scores(total_pred, total_label, test_loss, name):
    f1_fix = f1_score(total_pred, total_label, 0)*100
    f1_sacc = f1_score(total_pred, total_label, 1)*100
    f1_sp = f1_score(total_pred, total_label, 2)*100
    f1_blink = f1_score(total_pred, total_label, 3)*100
    f1_avg = (f1_fix + f1_sacc + f1_sp + f1_blink)/4
    print('\n{} set: Average loss: {:.4f}, F1_FIX: {:.2f}%, F1_SACC: {:.2f}%, F1_SP: {:.2f}%, F1_BLK: {:.2f}%, AVG: {:.2f}%\n'.format(
        name, test_loss, f1_fix, f1_sacc, f1_sp, f1_blink, f1_avg
    ))
    return (f1_fix + f1_sacc + f1_sp + f1_blink)/4


def set_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


def check_randomize(args, trX, trY):
    trX, trX_val = trX[:int(len(trX)*0.9)], trX[int(len(trX)*0.9):]
    trY, trY_val = trY[:int(len(trY)*0.9)], trY[int(len(trY)*0.9):] 
    if args.randomize and args.timesteps == 1:
        shuffler = np.random.permutation(len(trY))
        trX = trX[shuffler]
        trY = trY[shuffler]
    return trX, trY, trX_val, trY_val


def get_model(args, layers, features, n_classes):
    model = None
    if args.model == 'tcn':
        model = TCN(args.timesteps, n_classes, layers,
                    kernel_size=args.kernel_size, dropout=args.dropout)
    elif args.model == 'cnn_blstm':
        model = CNN_LSTM(args.timesteps, n_classes, args.kernel_size, args.dropout,
                          features, lstm_layers=2, bidirectional=True)
    elif args.model == 'cnn_lstm':
        model = CNN_LSTM(args.timesteps, n_classes, args.kernel_size, args.dropout,
                         features, lstm_layers=2)
    if model is not None:
        model.cuda()
    return model


def get_optimizer(args, model, learning_rate):
    if args.model.startswith('tcn'):
        return torch.optim.Adamax(model.parameters(), lr=learning_rate)
    else:
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)


def get_best_model(model, best_model, score, best_score):
    if score > best_score:
        print('>>> updating best model...')
        return copy.deepcopy(model), score
    return best_model, best_score




def train_OEMC(trainRecs, isolatedParticipant):
    pretrained_path = OEMC_MODEL
    set_randomness(0)
    args = OEMC_ArgsReplicator()
    folds = args.folds
    print("Loading data...")
    dataset = args.dataset
    freq = ET_SAMPLING_RATE
    pproc = preprocessor.Preprocessor(window_length=1,offset=args.offset,
                                      stride=args.strides,frequency=freq)
    # pproc.process_folder(INP_DIR, 'cached/VU', LABELER)
    
    fold = pproc.load_data_k_fold('cached/'+pproc.append_options(dataset), trainRecs, folds=folds)

    
    for fold_i in range(folds):
        _, _, trX, trY = next(fold)

        

        # trX, trX_val = trX[:int(len(trX)*0.9)], trX[int(len(trX)*0.9):]
        # trY, trY_val = trY[:int(len(trY)*0.9)], trY[int(len(trY)*0.9):] 
        train_size = len(trY)
        # val_size   = len(trY_val)
        n_classes  = 2
        features   = trX.shape[1]
        batch_size = args.batch_size
        epochs     = args.epochs
        channel_sizes = [30]*4
        timesteps  = args.timesteps
        rand = args.randomize
        scores = []
        steps = 0
        lr = args.lr

        # sanity check
        assert trY.max() < n_classes, (f"Label {trY.max().item()} invalid for n_classes={n_classes}")

        # load the pretrained model from the authors
        # model.load_state_dict(torch.load(pretrained_path))
        # model.cuda()
        pretrained_model = get_model(args, channel_sizes, features, n_classes=4)
        pretrained_model.load_state_dict(torch.load(pretrained_path))

        #initiate a new model for binary classification
        model = get_model(args, channel_sizes, features, n_classes)


        # Copy all weights except the last layer
        if args.model == 'tcn':
            # Assumes last layer is named `linear`
            model.load_state_dict({k: v for k, v in pretrained_model.state_dict().items() if 'linear' not in k}, strict=False)
            print(">>> Transferred all layers except final linear layer.")
        elif args.model in ['cnn_blstm', 'cnn_lstm']:
            # Adjust accordingly depending on how final layer is defined
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(">>> Transferred CNN-LSTM layers excluding final FC.")
            
        # best_model, best_score = None, 0
        optimizer = get_optimizer(args, model, lr)
        num_batches = train_size//batch_size
        # num_test_batches = val_size//batch_size

        print(f'>>> train size: {train_size}')
        print(f'>>> total batches: {num_batches}')

        for epoch in range(1, epochs+1):
            cost = 0
            for k in range(num_batches):
                start, end = k * batch_size, (k+1) * batch_size
                if start == 0:
                    start = timesteps
                trainX, trainY = pproc.create_batches(trX, trY, start, end, timesteps, rand) 
                cost += train(model, optimizer, trainX, trainY)
                steps += 1
                if k > 0 and (k % (num_batches//10) == 0 or k == num_batches-1):
                    print('Train Epoch: {:2} [{}/{} ({:.0f}%)]  Loss: {:.5f}  Steps: {}'.format(
                        epoch, end, train_size,
                        100*k/num_batches, cost/num_batches, steps 
                    ), end='\r')
            # t_loss, preds, labels = predict(model, num_batches, batch_size, 
            #                                 trX, trY, timesteps, pproc)
            # score = print_scores(preds, labels, t_loss, 'Val.')
            # scores.append(score)
            # if not args.no_lr_decay:
            #     if len(scores) >= 3 and (np.abs(scores[-1]-scores[-3]) < 0.1 
            #                          or scores[-1] < scores[-3]):
            #         lr /= 2
            #         print('[Epoch {}]: Updating learning rate to {:6f}\n'.format(epoch, lr))
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] = lr
            # best_model, best_score = get_best_model(model, best_model, score, best_score)
        
        # if not args.save_best:
        #     best_model = model

        #saving model and testing saved params
        model_param = "{}_model_{}_BATCH-{}_LOO-{}".format(
            args.model, dataset, batch_size, isolatedParticipant
        )
        if not os.path.exists('modules/methods/OEMC/models'):
            os.makedirs('modules/methods/OEMC/models')
        torch.save(model.state_dict(), 'modules/methods/OEMC/models/' + model_param + '.pt')