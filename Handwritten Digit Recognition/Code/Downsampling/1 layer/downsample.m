function [X_down] = downsample(inputImage, numrows)
    X_down = zeros(numrows,size(inputImage,2));
    for i = 1:size(inputImage,2)
        X_down(:,i) = imresize(inputImage(:,i),[numrows,1]);
    end
end