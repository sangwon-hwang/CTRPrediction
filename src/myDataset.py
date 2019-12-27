import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader


class myDataset(torch.utils.data.Dataset):
    def __init__(self, path, window=20, colNames=['Magnitude'], startIndex=0, endIndex=1000, stepSize=1):
        #         self.scaler = MinMaxScaler()
        self.path = path
        self.window = window  # look back window size
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.stepSize = stepSize
        self.colNames = colNames

        self.df = pd.read_csv(self.path, index_col='Time', usecols=colNames, na_values='nan')
        # self.df = pd.read_csv(self.path, usecols=['magnitude'], na_values='nan')
        self.df.fillna(method='ffill', inplace=True, axis=0)
        self.df.fillna(method='bfill', inplace=True, axis=0)
        self.trainData = self.df.as_matrix()
        #         self.df.numpyData = self.df.as_matrix(columns=['GOOGL'])
        #         self.trainData = self.scaler.fit_transform(self.numpyData)
        self.chunks = torch.FloatTensor(self.trainData[startIndex:endIndex]).unfold(0, self.window, self.stepSize).permute(0, 2, 1)

    def __len__(self):  # length of dataset
        return self.chunks.size(0)

    def __getitem__(self, index):  # 데이터 셋에서 한 개의 데이터를 가져오는 함수를 정의한다.
        x = self.chunks[index, :-1, :]
        y = self.chunks[index, -1, :]
        return x, y
