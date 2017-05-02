%Load variables
 load('norm_imgs_train.mat','norm_imgs_train');
 norm_imgs_train_karan = norm_imgs_train;
 load('labels_train.mat','labels_train');
 labels_train_karan = labels_train;
 
  load('norm_imgs_val.mat','norm_imgs_val');
 norm_imgs_val_karan = norm_imgs_val;
 load('labels_val.mat','labels_val');
 labels_val_karan = labels_val;
 
 load('norm_imgs_test.mat','norm_imgs_test');
 norm_imgs_test_karan = norm_imgs_test;
 load('labels_test.mat','labels_test');
 labels_test_karan = labels_test;

 %norm_imgs_karan = loadMNISTImages('train-images.idx3-ubyte');
 %   labels_karan = loadMNISTLabels('train-labels.idx1-ubyte');

% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(labels_train_karan, 1));
for n = 1: size(labels_train_karan, 1)
    targetValues(labels_train_karan(n) + 1, n) = 1;
end;

% Choose form of MLP:
numberOfHiddenUnits = 700;

% Choose appropriate parameters.
learningRate = 0.01;

% Choose activation function.
activationFunction = @ReLU;
dActivationFunction = @dReLU;

% Choose batch size and epochs. Remember there are 60k input values.
batchSize = 100;
epochs = 500;

fprintf('Train two layer perceptron with %d hidden units.\n', numberOfHiddenUnits);
fprintf('Learning rate: %d.\n', learningRate);

[hiddenWeights, outputWeights, error] = train_ReLU_1_layer(activationFunction, dActivationFunction, numberOfHiddenUnits, norm_imgs_train_karan, targetValues, epochs, batchSize, learningRate);

 %norm_imgs_test_karan = loadMNISTImages('t10k-images.idx3-ubyte');
 %   labels_test_karan = loadMNISTLabels('t10k-labels.idx1-ubyte');

 %-------------------------------------------------------------------
 fprintf('Validation Set:\n');

[correctlyClassified, classificationErrors] = val_ReLU_1_layer(activationFunction, hiddenWeights, outputWeights, norm_imgs_val_karan, labels_val_karan);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);
 
 %----------------------------------------------------------
% Choose decision rule.
fprintf('Test Set:\n');

[correctlyClassified, classificationErrors] = test_ReLU_1_layer(activationFunction, hiddenWeights, outputWeights, norm_imgs_test_karan, labels_test_karan);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);