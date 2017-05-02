function [hiddenWeights, outputWeights, error] = train_ReLU_1_layer(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
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
    
    %Dropout probability
    p = 0.5;

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);
    
    figure; hold on;

    for t = 1: epochs
        for k = 1: batchSize
            % Select which input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
           
            % Propagate the input vector through the network.
            inputVector = inputValues(:, n(k));  
            
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            hiddenOutputVectorSize = size(hiddenOutputVector,1);
            mask = ones(numberOfHiddenUnits,1);
            %Dropping out 20% units
            e = zeros(hiddenOutputVectorSize/5,1);
            for f=1:floor(hiddenOutputVectorSize/5)
                e(f) = floor(rand(1)*hiddenOutputVectorSize + 1);
                hiddenOutputVector(e(f),1) = 0;
                mask(e(f),1) = 0;
            end
            % Multiplying the hidden output vector by the probability
            %hiddenOutputVector = hiddenOutputVector.*p;
            
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDel = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDel = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDel);
            
            deltaoutput = learningRate.*outputDel*hiddenOutputVector';
            for k=1:size(e,1)
                deltaoutput(:,e(k)) = 0;
            end
            
            deltahidden = learningRate.*hiddenDel*inputVector';
            for l=1:size(e,1)
                deltahidden(e(l),:) = 0;
            end
            
            outputWeights = outputWeights - deltaoutput;
            hiddenWeights = hiddenWeights - deltahidden;
        end;
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = inputValues(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2);
        end;
        error = error/batchSize;
        
        plot(t, error,'*');
    end;
 end