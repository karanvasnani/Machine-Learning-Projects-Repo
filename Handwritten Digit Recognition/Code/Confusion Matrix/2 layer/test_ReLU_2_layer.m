function [correctlyClassified, classificationErrors] = test_ReLU_2_layer(activationFunction, hiddenWeights_1, hiddenWeights_2, outputWeights, inputValues, labels)
% validateTwoLayerPerceptron Validate the twolayer perceptron using the
% validation set.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% inputValues                    : Input values for training (784 x 10000).
% labels                         : Labels for validation (1 x 10000).
%
% OUTPUT:
% correctlyClassified            : Number of correctly classified values.
% classificationErrors           : Number of classification errors.
% 

    testSetSize = size(inputValues, 2);
    classificationErrors = 0;
    correctlyClassified = 0;
    confusion_mat = zeros(10,10);
    for n = 1: testSetSize
        inputVector = inputValues(:, n);
        outputVector = evaluateTwoLayerPerceptron(activationFunction, hiddenWeights_1, hiddenWeights_2, outputWeights, inputVector);
        
        class = decisionRule(outputVector);
        if class == labels(n) + 1
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end;
        %-------------------------------------------
        
        confusion_mat(labels(n)+1,class) = confusion_mat(labels(n)+1,class)+1;
        %-------------------------------------------
    end;
    confusion_mat
end

function class = decisionRule(outputVector)
% decisionRule Model based decision rule.
%
% INPUT:
% outputVector      : Output vector of the network.
%
% OUTPUT:
% class             : Class the vector is assigned to.
%

    max = 0;
    class = 1;
    for i = 1: size(outputVector, 1)
        if outputVector(i) > max
            max = outputVector(i);
            class = i;
        end;
    end;
end

function outputVector = evaluateTwoLayerPerceptron(activationFunction, hiddenWeights_1, hiddenWeights_2, outputWeights, inputVector)
% evaluateTwoLayerPerceptron Evaluate two-layer perceptron given by the
% weights using the given activation function.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% hiddenWeights                  : Weights of hidden layer.
% outputWeights                  : Weights for output layer.
% inputVector                    : Input vector to evaluate.
%
% OUTPUT:
% outputVector                   : Output of the perceptron.
% 

    outputVector = activationFunction(outputWeights*activationFunction(hiddenWeights_2*activationFunction(hiddenWeights_1*inputVector)));
end