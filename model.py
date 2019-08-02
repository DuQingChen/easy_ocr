import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(kernel_size=2,stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(kernel_size=2,stride=1)
		)

	def forward(self, x):
		b, channel, h, w = x.size()
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		# print(x.size())
		return x

class RNN(nn.Module):
    def __init__(self, class_num, hidden_unit):
        super(RNN, self).__init__()
        self.Bidirectional_LSTM1 = torch.nn.LSTM(128, hidden_unit, bidirectional=True)
        self.embedding1 = torch.nn.Linear(hidden_unit * 2, 128)
        self.Bidirectional_LSTM2 = torch.nn.LSTM(128, hidden_unit, bidirectional=True)
        self.embedding2 = torch.nn.Linear(hidden_unit * 2, class_num)

    def forward(self, x):
        x = self.Bidirectional_LSTM1(x)   # LSTM output: output, (h_n, c_n)
        T, b, h = x[0].size()   # x[0]: (seq_len, batch, num_directions * hidden_size)
        x = self.embedding1(x[0].view(T * b, h))  # pytorch view() reshape as [T * b, nOut]
        x = x.view(T, b, -1)  # [16, b, 512]
        x = self.Bidirectional_LSTM2(x)
        T, b, h = x[0].size()
        x = self.embedding2(x[0].view(T * b, h))
        x = x.view(T, b, -1)
        return x  # [16,b,class_num]		

class CRNN(nn.Module):
    def __init__(self, class_num, hidden_unit=128):
        super(CRNN, self).__init__()
        self.cnn = CNN()
        self.rnn = RNN(class_num, hidden_unit)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b,c,-1)
        # print(x.size)
        x = x.permute(2, 0, 1)  # [w, b, c] = [seq_len, batch, input_size]
        x = self.rnn(x)
        return x
if __name__ == '__main__':
	crnn = CRNN(37, 128)
	x = torch.zeros(1,1,35,90)
	output = crnn(x)
	print(output.size())