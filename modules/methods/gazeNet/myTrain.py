# train.py
import os
import sys
import time
import pickle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable


from modules.methods.gazeNet.utils_lib.data_loader import EMDataset, GazeDataLoader
from modules.methods.gazeNet.utils_lib import utils
from modules.methods.gazeNet.model import gazeNET as gazeNET
import modules.methods.gazeNet.model as model_func


def main(X_train, model_name, model_dir="my_model", num_epochs=10, num_workers=2, seed=123):
    # Set paths
    logdir = os.path.join("logdir", model_dir)
    config_path = os.path.join(logdir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    config = utils.Config(config_path).params
    config['split_seqs'] = True
    config['augment'] = False
    config['batch_size'] = 100
    
    # CUDA setup
    cuda = config['cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # # Load training and validation data
    # def load_pkl(filename):
    #     with open(os.path.join(logdir, "data", filename), 'rb') as f:
    #         return pickle.load(f)

    # X_train = load_pkl(config["data_train"][0])
    # X_val = load_pkl(config["data_val"][0])


    # Data loaders
    dataset_train = EMDataset(config=config, gaze_data=X_train)
    loader_train = GazeDataLoader(dataset_train, batch_size=config["batch_size"],
                                  num_workers=num_workers, shuffle=True, seed=seed)

    # dataset_val = EMDataset(config=config, gaze_data=X_val)
    # loader_val = GazeDataLoader(dataset_val, batch_size=1,
    #                             num_workers=num_workers, shuffle=False, seed=seed)

    # Load original 3-class model
    model = gazeNET(config, num_classes=3, seed=seed)
    model_func.load(model, "/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/gazeNet/logdir/model_final/models", config)
    model.to(device)

    # Replace final layer: 3 → 2 classes (reuse saccade weights)
    old_fc_layer = model.fc[0].module  # SequenceWise wraps a Linear layer
    new_fc_layer = torch.nn.Linear(old_fc_layer.in_features, 2, bias=False)

    with torch.no_grad():
        # Class 0 in your dataset = avg of old class 0 and 1
        new_fc_layer.weight[0] = 0.5 * (old_fc_layer.weight[0] + old_fc_layer.weight[1])
        # Class 1 in your dataset = old class 2 (saccades)
        new_fc_layer.weight[1] = old_fc_layer.weight[2]

    # Replace the layer inside the SequenceWise wrapper
    model.fc[0].module = new_fc_layer
    model = nn.DataParallel(model).to(device)
    model.train()

    # Optimizer and loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(logdir, "TB", "train_binary"))

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(tqdm(loader_train, desc=f"Epoch {epoch}")):
            inputs, targets, *_ = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            y_ = Variable(targets)
            
            outputs = model(inputs)

            yt, yn = outputs.size()[:2]
            y = outputs.view(yt * yn, -1)
            #WARNING (from the original author): only works for split_seqs=True;
            #i.e. all sequences need to be same exact length
            loss = criterion(y, y_)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader_train)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
    print("saving model "+ model_name)
    # torch.save(model.state_dict(), os.path.join(model_dir, model_name))

    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model

    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'config': config,  # this must be a JSON-serializable dictionary
        'epoch': epoch,    # optional
        'other_info': {...}  # anything else you want
    }

    torch.save(checkpoint, os.path.join(model_dir, model_name))
        # Optionally: add evaluation here

    writer.close()

if __name__ == "__main__":
    main()




            # y = outputs.view(-1, 2)
            # targets = targets.view(-1)

            # loss = criterion(y, targets)
            # optimizer.zero_grad()