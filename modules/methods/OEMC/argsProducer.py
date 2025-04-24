import argparse


def produceArgs():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d',
                           '--dataset',
                           required=False,
                           default= 'VU',
                           choices=['hmr', 'gazecom', 'VU'])
    argparser.add_argument('-m',
                           '--model',
                            required=False,
                            default= 'tcn',
                            choices=['tcn', 'cnn_blstm', 'cnn_lstm'])
    argparser.add_argument('-b',
                           '--batch_size',
                           required=False,
                           default=2048,
                           type=int)
    argparser.add_argument('--dropout',
                            required=False,
                            default=0.25,
                            type=float)
    argparser.add_argument('-e',
                           '--epochs',
                           required=False,
                           default=25,
                           type=int)
    argparser.add_argument('-k',
                           '--kernel_size',
                           required=False,
                           default=5,
                           type=int)
    argparser.add_argument('-t',
                           '--timesteps',
                           required=False,
                           default= 20, #TODO Check what it means
                           type=int)
    argparser.add_argument('-r',
                           '--randomize',
                            required=False,
                            action='store_true')
    argparser.add_argument('-f',
                           '--folds',
                           required=False,
                           default=1,
                        #    default=10,
                           type=int)
    argparser.add_argument('-s',
                           '--strides',
                           required=False,
                           default=9,
                           type=int)
    argparser.add_argument('-o',
                           '--offset',
                           required=False,
                           default=0,
                           type=int)
    argparser.add_argument('--lr',
                            required=False,
                            default=0.01,
                            type=float)
    argparser.add_argument('--no_lr_decay',
                           required=False,
                           action='store_true')
    argparser.add_argument('--save_best',
			               required=False,
                           action='store_true')
    argparser.add_argument('--out',
                            required=False,
                            default='final_outputs')
    argparser.add_argument('--mod',
                        required=False,
                        default='final_models')
    argparser.add_argument('--cpu',
                        action='store_true')
    args = argparser.parse_args()
    return args