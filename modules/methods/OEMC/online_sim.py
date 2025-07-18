import numpy as np
import preprocessor as pr
import argparse
from tcn import TCN
from cnn_lstm import CNN_LSTM
import torch
import os
import scorer
import time
from config import OEMC_MODEL

class OnlineSimulator():

    def __init__(self, args):
        self.args = args
        self.freq = 200 if args.dataset == 'hmr' else 250
        self.pr = pr.Preprocessor(window_length=1, 
                                  stride=args.strides, frequency=self.freq, offset=0)
        self._check_folder_exists(args.dataset)
        self.conf_matrix = {i:{'tp':0,'tn':0,'fp':0,'fn':0} for i in range(4)}
        self.conf_total = {'tp':0, 'fp':0, 'fn':0}
        self.scorer = scorer.Scorer(args)
        self.times = []
        torch.set_printoptions(sci_mode=False)


    def simulate(self, rec, model_dir, retrained, n_folds=None):
        fold = self.pr.load_data_k_fold('cached/'+self.pr.append_options(
                                        self.args.dataset), rec, 
                                        folds=self.args.folds)
        pred_all = []
        gt_all = []
        if n_folds is None:
            n_folds = self.args.folds
        for fold_i in range(n_folds):
            _, _, teX, teY = next(fold)
            features = teX.shape[1]
            model = self._load_model(features, fold_i+1, model_dir, retrained)
            times = []
            for i in range(len(teX)):
                if i >= self.args.timesteps:
                    window = self._fill_up_tensor(teX, i)
                    init_time = time.time()
                    pred = self._predict(model, window).cpu().numpy()[0][0]
                    pred_all.append(pred)   #addded to save the results
                    times.append(time.time() - init_time)
                    gt  = int(teY[i])   
                    gt_all.append(gt)       #addded to save the results
                    self._update_conf_matrix(pred, gt)
                    self._show_progress(i, teX, times)
            # print(f'\nFOLD {fold_i+1}\n------------------')
            #self.scorer._f_score_calc(self.conf_matrix, self.conf_total)
            self.times += times
            # self._show_times(self.times)
        return pred_all, gt_all


    def OEMC_pred(self, model_dir, retrained):
        fold = self.pr.load_data_k_fold('cached/'+self.pr.append_options(
                                        self.args.dataset), 
                                        folds=self.args.folds)
        _, _, teX, teY = next(fold)
        features = teX.shape[1]
        # teX = self._fill_up_tensor(teX)
        teX = torch.from_numpy(teX).float()
        model = self._load_model(features, 1+1, model_dir, retrained)
        pred = self._predict(model, teX).cpu().numpy()[0][0]
        return pred

    def _check_folder_exists(self, dataset):
        if not os.path.exists("cached/" + self.pr.append_options(dataset)):
            if dataset == 'hmr':
                # self.pr.process_folder_parallel('data_hmr','cached/hmr', workers=12)
                self.pr.process_folder('data_hmr','cached/hmr')
            elif dataset == 'VU':
                self.pr.process_folder('/media/ash/Expansion/data/Saccade-Detection-Methods','cached/vu')
            elif dataset == 'gazecom':
                self.pr.process_folder_parallel('data_gazecom','cached/gazecom', workers=12)


    def _show_progress(self, i, data, times):
        if i % 1000 == 0:
            porc = i/len(data) * 100
            t = np.mean(times) * 1000
            print("Progress: {:2.3f}%  Avg time: {:1.4f}ms".format(porc,t), end='\r', flush=True)
                

    def _show_times(self, times):
        median = np.median(times)*1000
        mean = np.mean(times)*1000
        std = np.std(times)*1000
        csv = f"median,mean,std\n{median},{mean},{std}"
        print('Median time: {:1.4f} ms'.format(median))
        print('Mean time: {:1.4f} ms'.format(mean))
        print('Standard deviation: {:1.4f} ms'.format(std))
        # with open(f'latency_{self.args.model}_{self.args.dataset}.csv', 'w') as f:
        #     f.write(csv)


    def _load_model(self, features, fold, model_dir, retrained):
        filename = f"{self.args.model}_model_{self.args.dataset}_BATCH-"
        filename += f"{self.args.batch_size}_EPOCHS-{self.args.epochs}_FOLD-"
        filename += f"{fold}.pt"
        # path = os.path.join(self.args.mod, filename)
        path = model_dir
        if retrained:
            num_classes = 2
        else:
            num_classes = 4
        if self.args.model == 'tcn':
            model = TCN(self.args.timesteps, num_classes, [30]*4,
                kernel_size=self.args.kernel_size, dropout=self.args.dropout)
        elif self.args.model == 'cnn_lstm':
            model = CNN_LSTM(self.args.timesteps, num_classes, self.args.kernel_size, 
                 self.args.dropout, features, self.args.lstm_layers)
        else:
            model = CNN_LSTM(self.args.timesteps, num_classes, self.args.kernel_size,
                 self.args.dropout, features, self.args.lstm_layers,
                 bidirectional=True)
        if self.args.cpu:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path))
            model.cuda()
        self.print_parameters(model)
        model.eval()
        return model


    def _predict(self, model, sample):
        with torch.no_grad():
            sample = torch.autograd.Variable(sample, requires_grad=False)
            pred = model(sample)
            layer = torch.nn.Softmax(dim=1)
            pred = layer(pred)
            pred = pred.data.max(1, keepdim=True)[1]
            return pred

    
    def _fill_up_tensor(self, X, i):
        t = self.args.timesteps
        sample = np.array([X[i-t:i]])
        sample = torch.from_numpy(sample).float()
        if not self.args.cpu:
            return sample.cuda()
        return sample


    def _convert_label(self, target):
        if target == 0:
            return 'F'
        if target == 1:
            return 'S'
        if target == 2:
            return 'P'
        if target == 3:
            return 'B'

    
    def print_parameters(self, model):
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        print('>>> Trainable parameters:', params)



    def _update_conf_matrix(self, pred, gt):
        patterns = {0,1,2,3}
        if pred == gt:
            patterns.remove(gt)
            self.conf_matrix[gt]['tp'] += 1
            for i in patterns:
                self.conf_matrix[i]['tn'] += 1
        elif pred != gt:
            self.conf_matrix[gt]['fn'] += 1
            patterns.remove(gt)
            self.conf_matrix[pred]['fp'] += 1
            patterns.remove(pred)
            for i in patterns:
                self.conf_matrix[i]['tn'] += 1



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        default= 'tcn',
                        required=False)
    parser.add_argument('-d',
                        '--dataset',
                        required=False,
                        default= 'VU')
    parser.add_argument('-b',
                        '--batch_size',
                        required=False,
                        default=2048,
                        type=int)
    parser.add_argument('-e',
                        '--epochs',
                        required=False,
                        default=25,
                        type=int)
    parser.add_argument('-t',
                        '--timesteps',
                        required=False,
                        default= 20, #TODO Check what it means
                        type=int)
    parser.add_argument('-k',
                        '--kernel_size',
                        required=False,
                        default=5,
                        type=int)
    parser.add_argument('--dropout',
                        required=False,
                        default=0.25,
                        type=float)
    parser.add_argument('--lstm_layers',
                        required=False,
                        default=2,
                        type=int)
    parser.add_argument('-f',
                        '--folds',
                        required=False,
                        default=5,
                        type=int)
    parser.add_argument('-s',
                        '--strides',
                        required=False,
                        default=8,
                        type=int),
    parser.add_argument('--cpu',
                        action='store_true')
    parser.add_argument('--out',
                        required=False,
                        default='final_outputs')
    parser.add_argument('--mod',
                        required=False,
                        default='final_models')
    args = parser.parse_args()
    online_sim = OnlineSimulator(args)
    online_sim.simulate(1)
