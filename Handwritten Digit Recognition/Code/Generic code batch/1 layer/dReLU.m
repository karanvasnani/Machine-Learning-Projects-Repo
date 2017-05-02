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
    inputSize_x = size(x,1);
    inputSize_y = size(x,2);
    y = zeros(inputSize_x,inputSize_y);
    for j=1:inputSize_y
        for i=1:inputSize_x
            if x(i,j) > 0
                y(i,j) = 1;
            end
        end
    end
end