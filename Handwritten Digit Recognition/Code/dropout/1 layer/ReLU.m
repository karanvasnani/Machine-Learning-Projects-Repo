function y = ReLU(x)
% simpleLogisticSigmoid Logistic sigmoid activation function
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the logistic sigmoid was applied element by
% element.
%
    inputSize = size(x,1);
    y = zeros(inputSize,1);
    for i=1:inputSize
        if x(i,1) > 0
            y(i,1) = x(i,1);
        end
    end
end