%Load variables
 load('norm_imgs_train.mat','norm_imgs_train');
 %norm_imgs_train_karan = norm_imgs_train;
 [EigenVec,norm_imgs_train_karan_mean,norm_imgs_train_karan] = pca(norm_imgs_train);
 load('labels_train.mat','labels_train');
 labels_train_karan = labels_train;
 
 %-----Project the validation set on the obtained eigen vectors
 load('norm_imgs_val.mat','norm_imgs_val');
 %norm_imgs_val_karan = norm_imgs_val;
 load('labels_val.mat','labels_val');
 labels_val_karan = labels_val;
 
 X_val_dev = zeros(784,size(norm_imgs_val,2));
    for q=1:size(norm_imgs_val,2)
        X_val_dev(:,q) = (norm_imgs_val(:,q)) - (norm_imgs_train_karan_mean(:,1));
    end

%Projection of each image onto the eigen vectors
norm_imgs_val_karan = transpose(EigenVec)*X_val_dev;
 
%-----Project the test set on the obtained eigen vectors
 load('norm_imgs_test.mat','norm_imgs_test');
 %norm_imgs_test_karan = norm_imgs_test;
 load('labels_test.mat','labels_test');
 labels_test_karan = labels_test;
 
 X_test_dev = zeros(784,size(norm_imgs_test,2));
    for q=1:size(norm_imgs_test,2)
        X_test_dev(:,q) = (norm_imgs_test(:,q)) - (norm_imgs_train_karan_mean(:,1));
    end

%Projection of each image onto the eigen vectors
norm_imgs_test_karan = transpose(EigenVec)*X_test_dev;

 %norm_imgs_karan = loadMNISTImages('train-images.idx3-ubyte');
 %labels_karan = loadMNISTLabels('train-labels.idx1-ubyte');

% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(labels_train_karan, 1));
for n = 1: size(labels_train_karan, 1)
    targetValues(labels_train_karan(n) + 1, n) = 1;
end;

% Choose form of MLP:
hidden_units_layer_1 = 700;
hidden_units_layer_2 = 700;

% Choose appropriate parameters.
learningRate = 0.01;

% Choose activation function.
activationFunction = @ReLU;
dActivationFunction = @dReLU;

% Choose batch size and epochs. Remember there are 60k input values.
batchSize = 100;
epochs = 500;

fprintf('Layers = 3 | Layer 1 units = %d | Layer 2 units = %d\n', hidden_units_layer_1,hidden_units_layer_2);
fprintf('Learning rate: %d.\n', learningRate);

[hiddenWeights_1, hiddenWeights_2, outputWeights, error] = train_ReLU_2_layer(activationFunction, dActivationFunction, hidden_units_layer_1, hidden_units_layer_2, norm_imgs_train_karan, targetValues, epochs, batchSize, learningRate);

 %norm_imgs_test_karan = loadMNISTImages('t10k-images.idx3-ubyte');
 %   labels_test_karan = loadMNISTLabels('t10k-labels.idx1-ubyte');

 %-------------------------------------------------------------------
 fprintf('Validation Set:\n');

[correctlyClassified, classificationErrors] = val_ReLU_2_layer(activationFunction, hiddenWeights_1, hiddenWeights_2, outputWeights, norm_imgs_val_karan, labels_val_karan);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);
 
 %----------------------------------------------------------
% Choose decision rule.
fprintf('Test Set:\n');

[correctlyClassified, classificationErrors] = test_ReLU_2_layer(activationFunction, hiddenWeights_1, hiddenWeights_2, outputWeights, norm_imgs_test_karan, labels_test_karan);

fprintf('Classification errors: %d\n', classificationErrors);
fprintf('Correctly classified: %d\n', correctlyClassified);