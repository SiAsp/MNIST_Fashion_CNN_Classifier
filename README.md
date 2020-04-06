INFO284_Asignment2
sas020

This project is a Convolutional Neural Network achieving an accuracy of 92.120% on the MNIST Fashion dataset.


Comments on hyperparameters:

The network is defined with a learningrate (lr) of 0.001, 30 epochs and a batch size of 100.

Through testing I found that increasing the lr would result in the network missing the bottom of the learning-slope, and decreasing the lr resulted in the network not learning fast enough. The model was tested with other lrs from 0.0001 to 0.003.

With the current layering of the model I have found that around 30 epochs are needed for it to close in on it's best possible accuracy. The model was also tested with 40 epochs, this resulted in a testing accuracy of 91.7, but the delta between training and testing sets also increased. The training accuracy on the last epoch was measured at 96.1.

The batch size of 100 is the highest I was able to use, before the model would start to overfit, this could be spotted in the acc_history graph, where the accuracy would drastically drop at the start of each epoch. In some instances the accuracy would even drop from ~90% to ~40%. Through my testing the highest possible batch size, where the model did not overfit, yielded the best testing accuracy.

There is still some overfitting in the model, the delta between the training and test sets is ~3%.

I played around with different layering of the model, and found this to be the optimal. Removing the dropout results in the model overfitting on the training data, and other channel-selections, either more or less, would result in a lower testing accuracy, but keeping the slight overfitting as is now. The same was the situation for the fully connected layer, where more layers would decrease the testing accuracy, while keeping or increasing the slight overfitting.
