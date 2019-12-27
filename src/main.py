from itertools import repeat
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter

from models.CNNz import NP1DCNN
from myDataset import myDataset

if __name__ == "__main__":
    # Use cuda if available
    use_cuda = torch.cuda.is_available()
    print("use_cuda = ", use_cuda)
    # Keep track of loss in tensorboard
    writer = SummaryWriter()

    # Hyper Parameters for NN
    learning_rate = 0.001
    batch_size = 16
    # display_step = 200
    max_epochs = 15
    n_hidden1 = 128
    n_hidden2 = 128
    stepSize = 1
    weight_decay = 0.01

    # Parameters for myDataset
    window = 20
    startIndex = 0
    endIndex = 1000

    #     fn_base = "_epochs_" + str(max_epochs) + "_window_" + str(window) + "_weight_decay_" + \
    #               str(weight_decay) + "_train_" + startIndex + \
    #               "_" + endIndex

    path = '../data/cvr.csv'
    # path = '../data/click.csv'
    colNames = ['CVR', 'Click']  # ['Magnitude']
    # n_stocks = len(colNames)
    # n_output = n_stocks

    dset = myDataset(path,
                     window=window,
                     startIndex=startIndex,
                     endIndex=endIndex,
                     stepSize=stepSize,
                     colNames=colNames,
                     )

    trainData = DataLoader(dset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=4,
                           # pin_memory=use_cuda
                           )

    model = NP1DCNN(window=window, hidden_size=64, dilation=2)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.MSELoss(size_average=False)  # .cuda()

    losses = []
    for i in range(max_epochs):
        loss_ = 0.
        predicted = []
        gt = []
        # Go through training data set
        for batch_idx, (data, target) in enumerate(trainData):
            data = Variable(data.permute(0, 2, 1)).unsqueeze_(1).contiguous()
            target = Variable(target.unsqueeze_(1))
            # if use_cuda:
            #     data = data.cuda()
            #     target = target.cuda()
            if target.data.size()[0] == batch_size:
                optimizer.zero_grad()   # set weights to 0
                output = model(data)
                loss = criterion(output, target)
                loss_ += loss.data
                loss.backward()         # Backpropagation
                optimizer.step()        # Gradient descent
                # Store current results for visual check
                for k in range(batch_size):
                    predicted.append(output.data[k, 0, :].cpu().numpy())
                    gt.append(target.data[k, 0, :].cpu().numpy())

        print("Epoch = ", i)
        print("Loss = ", loss_)
        losses.append(loss_)

        # Store for display in Tensorboard
        writer.add_scalar("loss_epoch", loss_, i)
        # Apply step of scheduler for learning rate change
        scheduler_model.step()

        if i == (max_epochs-1):
            predicted = np.array(predicted)
            gt = np.array(gt)
            x = np.array(range(predicted.shape[0]))
            plt.figure(num=0, figsize=[8, 6])
            plt.plot(x, gt[:, 0], label="raw", color=cm.Blues(100))              # , color=cm.Greens(50)
            plt.plot(x, predicted[:, 0], "ro", label="predicted", color=cm.Blues(250)) # , color=cm.Reds(150)
            plt.legend()
            plt.show()

    # Save trained models
    # torch.save(model, 'conv2d_' + fn_base + '.pkl')
    # Plot training loss
    h = plt.figure(num=1, figsize=[8, 6])
    x = range(len(losses))
    plt.plot(np.array(x), np.array(losses), label="loss", color=cm.Blues(250))   # , color=cm.autumn(200)
    plt.xlabel("Time")
    plt.ylabel("Training loss")
    # plt.savefig("loss_" + fn_base + '.png')
    plt.legend()
    plt.show()


    # Test
    # path = '../data/GOOGL_data.csv'
    # # path = '/Users/sangwonhwang/Desktop/03.CTRPrediction/data/cvr.csv'
    dtest = myDataset(
        path,
        window=window,
        startIndex=startIndex,
        endIndex=endIndex,
        stepSize=stepSize
    )

    trainData = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=use_cuda
    )

    # TEST
    n_stocks = 1
    predictions = np.zeros((len(trainData.dataset.chunks), n_stocks))  # n_stocks is len(symbols)
    ground_tr = np.zeros((len(trainData.dataset.chunks), n_stocks))  #
    batch_size_pred = batch_size

    # Create list of n_stocks lists for storing predictions and GT
    predictions = [[]]
    gts = [[]]
    k = 0

    symbols = ['GOOGL']
    # Predictions
    for batch_idx, (data, target) in enumerate(trainData):
        data = Variable(data.permute(0, 2, 1)).unsqueeze_(1).contiguous()
        target = Variable(target.unsqueeze_(1))
        # if use_cuda:
        #     data = data.cuda()
        #     target = target.cuda()
        k = 0

        if target.data.size()[0] == batch_size_pred:
            output = model(data) # In this time, we do not call creator function for model object

            for i in range(batch_size_pred):
                s = 0
                for stock in symbols:
                    predictions[s].append(output.data[i, 0, s])
                    gts[s].append(target.data[i, 0, s])
                    s += 1
                k += 1

    # Plot results
    # Convert lists to np array for plot, and rescaling to original data
    if len(symbols) == 1:
        #     pred = dtest.scaler.inverse_transform(np.array(predictions[0]).reshape((len(predictions[0]), 1)))
        #     gt = dtest.scaler.inverse_transform(np.array(gts[0]).reshape(len(gts[0]), 1))
        pred = np.array(predictions[0]).reshape((len(predictions[0]), 1))
        gt = np.array(gts[0]).reshape(len(gts[0]), 1)

    if len(symbols) >= 2:
        p = np.array(predictions)
        pred = dtest.scaler.inverse_transform(np.array(predictions).transpose())
        gt = dtest.scaler.inverse_transform(np.array(gts).transpose())
    # Plot for all stocks in
    # x = [np.datetime64(start_date) + np.timedelta64(x, 'D') for x in range(0, pred.shape[0])]
    # x = np.array(x)
    # ut = len(trainData.dataset.chunks)
    s = 0
    # for stock in symbols:

    plt.figure(num=2, figsize=[8, 6])
    temp_y = pred[:, s]
    temp_y = len(temp_y)
    x = list(range(temp_y))
    plt.plot(x, gt[:, s], label="raw", color=cm.Blues(100))                    #, color=cm.Greens(50)
    plt.plot(x, pred[:, s], ls="--", label="predicted", color=cm.Blues(250))   #, color=cm.Reds(100)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()

    # # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_major_formatter(monthsFmt)
    # plt.title(stock)
    # fig.autofmt_xdate()
    # #     plt.savefig(stock + "_" + fn_base + '.png')
    # plt.show()
    s += 1





