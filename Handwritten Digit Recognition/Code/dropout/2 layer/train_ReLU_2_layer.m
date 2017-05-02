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

    %Dropout probability
    p = 0.5;

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
            
            hidden_1_OutputVectorSize = size(hidden_1_OutputVector,1);
            mask_1 = ones(hidden_units_layer_1,1);
            %Dropping out 20% hidden layer 1 units
            e = zeros(hidden_1_OutputVectorSize/5,1);
            for f=1:floor(hidden_1_OutputVectorSize/5)
                e(f) = floor(rand(1)*hidden_1_OutputVectorSize + 1);
                hidden_1_OutputVector(e(f),1) = 0;
                mask_1(e(f),1) = 0;
            end
            % Multiplying the hidden output vector by the probability
            %hidden_1_OutputVector = hidden_1_OutputVector.*p;
            
            hidden_2_ActualInput = hiddenWeights_2*hidden_1_OutputVector;
            hidden_2_OutputVector = activationFunction(hidden_2_ActualInput);
            
            hidden_2_OutputVectorSize = size(hidden_2_OutputVector,1);
            mask_2 = ones(hidden_units_layer_2,1);
            %Dropping out 20% hidden layer 2 units
            u = zeros(hidden_2_OutputVectorSize/5,1);
            for f=1:floor(hidden_2_OutputVectorSize/5)
                u(f) = floor(rand(1)*hidden_2_OutputVectorSize + 1);
                hidden_2_OutputVector(u(f),1) = 0;
                mask_2(u(f),1) = 0;
            end
            % Multiplying the hidden output vector by the probability
            %hidden_2_OutputVector = hidden_2_OutputVector.*p;
            
            outputActualInput = outputWeights*hidden_2_OutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDel = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDel_2 = dActivationFunction(hidden_2_ActualInput).*(outputWeights'*outputDel);
            hiddenDel_1 = dActivationFunction(hidden_1_ActualInput).*(hiddenWeights_2'*hiddenDel_2);
            
            deltaoutput = learningRate.*outputDel*hidden_2_OutputVector';
            for k=1:size(u,1)
                deltaoutput(:,u(k)) = 0;        %removing columns corres to u
            end
            
            deltahidden_2 = learningRate.*hiddenDel_2*hidden_1_OutputVector';
            for l=1:size(u,1)
                deltahidden_2(u(l),:) = 0;        %removing rows corres to u
            end
            
            for m=1:size(e,1)
                deltahidden_2(:,e(m)) = 0;        %removing columns corres to e
            end
            
            deltahidden_1 = learningRate.*hiddenDel_1*inputVector';
            for h=1:size(e,1)
                deltahidden(e(h),:) = 0;        %removing rows corres to e
            end
            
            outputWeights = outputWeights - deltaoutput;
            hiddenWeights_2 = hiddenWeights_2 - deltahidden_2;
            hiddenWeights_1 = hiddenWeights_1 - deltahidden_1;
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