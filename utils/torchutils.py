# -*- coding: utf-8 -*-
#
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, device, trainset, valset,
                criterion, optimizer, epochs=10, batch_size=32):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_batches = np.ceil(trainset.num_examples / batch_size).astype('int')
    val_batches = np.ceil(valset.num_examples / batch_size).astype('int')

    for epoch in range(epochs):
        loss_train = 0
        loss_val = 0
        acc_train = 0.
        acc_val = 0.

        print("Epoch {}/{}".format(epoch, epochs))
        print('-' * 10)

        model.train(True)
        for step in range(train_batches):
            X_batch, y_batch = trainset.next_batch(batch_size)
            X_batch = torch.tensor(X_batch, device=device)
            y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            acc_train += torch.sum(preds == y_batch.data)

            del X_batch, y_batch, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_train / trainset.num_examples
        avg_acc = acc_train / trainset.num_examples

        model.train(False)
        model.eval()
        for step in range(val_batches):
            X_batch, y_batch = valset.next_batch(batch_size)
            X_batch = torch.tensor(X_batch, device=device)
            y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, y_batch)

            loss_val += loss.item()
            acc_val += torch.sum(preds == y_batch.data)

            del X_batch, y_batch, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / valset.num_examples
        avg_acc_val = acc_val / valset.num_examples

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model