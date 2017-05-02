function y = dReLU(x)
% Derivative of the dReLU.
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the derivative of the ReLU was
% applied element by element.
%
    inputSize = size(x,1);
    y = zeros(inputSize,1);
    for i=1:inputSize
        if x(i,1) > 0
            y(i,1) = 1;
        end
    end
end