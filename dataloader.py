import numpy as np
import cv2
import os
import dictionary

def normal(img):
	img = img/127.5
	img = img - 1
	return img

def dataloader(batch_size = 8, root = 'test_img/'):
	roots = os.listdir(root)
	for r in roots: #  二级目录
		length = int(r) # 文件夹名就是输出长度
		r = root+r+'/'
		paths = os.listdir(r)
		batches = len(paths) // batch_size
		for batch_i in range(batches):
			imgs, targets = np.zeros([batch_size,1,35,90]), np.zeros([batch_size, length])
			tmp_paths = paths[batch_i*batch_size:(batch_i+1)*batch_size]
			for i, p in enumerate(tmp_paths):
				target = p.split('_')[0] # 文件名包含文本信息
				img = cv2.imread(r+p, 0)
				imgs[i,:], targets[i] = normal(img), dictionary.decode(target)
			yield imgs, targets

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	loader = dataloader()
	for i in range(1):
		imgs, targets = loader.__next__()
	print(imgs.shape, targets.shape)
	b, l = targets.shape
	for i in range(b):
		img, target = imgs[i,0], targets[i]
		print(dictionary.encode(target))
		plt.imshow(img, cmap='gray')
		plt.show()
	
		

