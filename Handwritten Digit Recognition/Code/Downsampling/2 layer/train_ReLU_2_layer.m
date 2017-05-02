function [hiddenWeights_1, hiddenWeights_2, outputWeights, error] = train_ReLU_2_layer(activationFunction, dActivationFunction, hidden_units_layer_1, hidden_units_layer_2, inputValues, targetValues, epochs, batchSize, learningRate)
% trainReLUPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights_1 = rand(hidden_units_layer_1, inputDimensions);
    hiddenWeights_2 = rand(hidden_units_layer_2, hidden_units_layer_1);
    outputWeights = rand(outputDimensions, hidden_units_layer_2);
    
    hiddenWeights_1 = hiddenWeights_1./size(hiddenWeights_1, 2);
    hiddenWeights_2 = hiddenWeights_2./size(hiddenWeights_2, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);
    
    figure; hold on;

    for t = 1: epochs
        for k = 1: batchSize
            % Select which input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k));
            hidden_1_ActualInput = hiddenWeights_1*inputVector;
            hidden_1_OutputVector = activationFunction(hidden_1_ActualInput);
            hidden_2_ActualInput = hiddenWeights_2*hidden_1_OutputVector;
            hidden_2_OutputVector = activationFunction(hidden_2_ActualInput);
            outputActualInput = outputWeights*hidden_2_OutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta_2 = dActivationFunction(hidden_2_ActualInput).*(outputWeights'*outputDelta);
            hiddenDelta_1 = dActivationFunction(hidden_1_ActualInput).*(hiddenWeights_2'*hiddenDelta_2);
            
            outputWeights = outputWeights - learningRate.*outputDelta*hidden_2_OutputVector';
            hiddenWeights_2 = hiddenWeights_2 - learningRate.*hiddenDelta_2*hidden_1_OutputVector';
            hiddenWeights_1 = hiddenWeights_1 - learningRate.*hiddenDelta_1*inputVector';
        end;
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights_2*activationFunction(hiddenWeights_1*inputVector))) - targetVector, 2);
        end;
        error = error/batchSize;
        
        plot(t, error,'*');
    end;
 end