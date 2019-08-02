import numpy as np

num2str = list(' abcdefghijklmnopqrstuvwxyz0123456789')
str2num = dict(zip(num2str, range(37)))
def decode(strs):
	strs = list(strs)
	length = len(strs)
	data = np.zeros([length,])
	for i, s in enumerate(strs):
		data[i] = str2num[s]
	return data

def encode(nums):
	length = list(nums)
	data = []
	for i, num in enumerate(nums):
		data.append(num2str[int(num)])
	return ''.join(data)

if __name__ == '__main__':
	data = decode('abc')
	print(data, type(data))
	data = encode(data)
	print(data, type(data))
# print(str2num)