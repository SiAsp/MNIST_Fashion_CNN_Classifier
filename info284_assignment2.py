import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from torch.autograd import Variable
from torch.nn import Linear, Module, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from sklearn.metrics import classification_report


class CNN(Module):
	'''Convolutional Neural Network model.'''
	def __init__(self, lr, epochs, batch_size, num_classes = 10):
		super(CNN, self).__init__()

		self.lr = lr
		self.epochs = epochs
		self.batch_size = batch_size
		self.num_classes = num_classes
		self.loss_history = []
		self.acc_history = []
		self.predictions = []
		self.target = []
		self.class_labels = [
			'T-shirt/top'
			'Trouser',
			'Pullover',
			'Dress',
			'Coat',
			'Sandal',
			'Shirt',
			'Sneaker',
			'Bag',
			'Ankle boot',
		]

		self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

		self.convolutional_layers = Sequential(	
					
			Conv2d(1, 64, 3),
			Dropout(0.5),
			BatchNorm2d(64),
			ReLU(),

			Conv2d(64, 256, 3),
			Dropout(0.5),
			BatchNorm2d(256),
			ReLU(),
			
			MaxPool2d(2),

			Conv2d(256, 256, 3),
			Dropout(0.5),
			BatchNorm2d(256),
			ReLU(),

			Conv2d(256, 256, 3),
			Dropout(0.5),
			BatchNorm2d(256),
			ReLU(),
			
			MaxPool2d(2),
		)

		# Calculate input-dimentions for fully connected layer
		input_dims = self.calc_input_dims()
		self.fc = Sequential(
			Linear(input_dims, self.num_classes),
		)

		self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
		self.criterion = CrossEntropyLoss()

		self.to(self.device)
		self.get_data()

	def calc_input_dims(self):
		'''Calculate input-dimentions for fully connected layer'''
		batch_data = T.zeros((1, 1, 28, 28))
		return int(np.prod(self.convolutional_layers(batch_data).size()))

	def get_data(self):
		'''Download and load training and test data'''
		## Transformations to tensor
		transform = transforms.Compose([transforms.ToTensor()])

		## Download and load training dataset
		self.trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
		self.trainloader = T.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,shuffle=True, num_workers=12)

		## Download and load testing dataset
		self.testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform)
		self.testloader = T.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=12)

	def forward(self, batch_data):
		'''Forward propagation step'''
		batch_data = T.tensor(batch_data).to(self.device)

		batch_data = self.convolutional_layers(batch_data)
		batch_data = batch_data.flatten(start_dim = 1)

		classes = self.fc(batch_data)
		return classes

	def get_accuracy(self, logit, target):
		''' Obtain accuracy for training round '''
		corrects = (logit.view(target.size()).data == target.data).sum()
		accuracy = 100.0 * corrects / self.batch_size
		return accuracy.item()


	def _train(self):
		'''Train the model'''
		self.train()
		for i in range(self.epochs):
			train_loss = 0
			train_acc = []
			for j, (images, labels) in enumerate(self.trainloader):
				self.optimizer.zero_grad()

				labels = labels.to(self.device)

				# Forward + backprop + loss
				logits = self.forward(images)
				loss = self.criterion(logits, labels)
				logits = F.softmax(logits, dim=1).argmax(dim=1)

				prediction = self.get_accuracy(logits, labels)

				train_loss += loss.detach().item()
				train_acc.append(prediction)
				self.acc_history.append(prediction)

				loss.backward()
				self.optimizer.step()

			self.loss_history.append(train_loss)
			print(f'Epoch: {i} | Loss: {train_loss / j:.4f} | Train Accuracy: {np.mean(train_acc):.3f}')

	def test(self):
		'''Test the model'''
		test_loss = 0
		test_acc = []
		for i, (images, labels) in enumerate(self.testloader):
			labels = labels.to(self.device)

			# Forward + backprop + loss
			logits = self.forward(images)
			loss = self.criterion(logits, labels)
			logits = F.softmax(logits, dim=1).argmax(dim=1)
			self.predictions.extend(logits)
			self.target.extend(labels)

			prediction = self.get_accuracy(logits, labels)

			test_loss += loss.detach().item()
			test_acc.append(prediction)

		print(f'Loss: {test_loss / i:.4f} | Accuracy: {np.mean(test_acc):.3f}')

	def create_report(self):
		# Print classification report
		classicifaction_metrix = classification_report(self.target, self.predictions, target_names=range(9))
		print('Classification Report: \n', classification_report)

		# Plot testing loss and accuracy gradient
		plt.plot(net.loss_history)
		plt.show()
		plt.plot(net.acc_history)
		plt.show()

		# Create confusion matrix
		conf_matr = confusion_matrix(self.target, self.predictions)
		plt.matshow(conf_matr)


if __name__ == "__main__":
	net = CNN(lr=0.001, epochs=30, batch_size=100)
	net._train()
	net.test()
	net.create_report()

	if input('Save model? y | n ').lower() == 'y':
		T.save(net, './fashionMNIST_classifier.pt')