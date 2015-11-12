# BirdID_CNN


--- STRUCTURE ---

Classify a set of 9 category bird images using convolutionary neural network with Lasagne API

Convolutionary Neural Network (CNN) were used to classify photos of 9 species of birds. The dataset had a minimum of 98 images per category.

Images are resized to 140x140, and then augmented using random horizontal flips and crops to 128x128 with random offsets. The validation set goes through the exact same method for augmentation.

The networks were trained using stochastic gradient descent(SGD), utilizing the adaptive subgradient method (Adagrad) to change the learning rate over time. The initial learning rate with adagrad was set to 0.01.

An L2 regularization penalty was applied to the stoachastic loss function to allow for better generalization (l2 regularization rate = 0.0001)

Rectified linear units were used as the activation function for both the convolutional and fully connected layers.

"Same" convolutions were used through zero-padding to keep the input and output dimensions the same.






--- RESULT ---

Case 1: Classifying 2 categories of birds 
        (Carolina Wren && Tufted Titmouse)
Case 2: Classifying 9 categories of birds 
        (Caroline Chickadee && Crolina Wren && Downy Wood Pecker && Northern Cardinal && Red Bellied Woodpe && Tufted Titmouse && White Throated Spar && Yellow Rumped Warbler)

The CNN script and result can be found in folder "./Nov2nd2015"
