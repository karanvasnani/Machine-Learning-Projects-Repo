function [V,X_mean,X_database] = pca(X)
%  load('norm_imgs_train.mat','norm_imgs_train');
%  X = norm_imgs_train;
%  load('labels_train.mat','labels_train');
%  labels_train_karan = labels_train;

 K=100;      %no of eigen vectors to be considered
%for d=1:13
    %------------PCA on GallerySet----------------

    %Finding the mean face by taking average of all image vectors
    X_mean = zeros(784,1);
    for k=1:784
        X_mean(k,1) = mean(X(k,:));
    end

    %Converting mean face vector to matrix for display purpose
%     X_mean_face = zeros(28,28,'uint8');
%     for p=1:28
%         X_mean_face(:,p) = X_mean((28*(p-1)+1):28*p,1);
%     end

    %Deviation of each image from the mean
    X_dev = zeros(784,size(X,2));
    for q=1:size(X,2)
        X_dev(:,q) = (X(:,q)) - (X_mean(:,1));
    end

    %Covariance matrix
    C = X_dev*transpose(X_dev);

    %finding K largest eigenvalues and corresponding eigenvectors of L matrix
    [V,l] = eigs(C,K);
    %U = transpose(X_dev)*V;
    % 3 Largest Eigen Faces
    % eigen_face_1 = zeros(50,50,'uint8');
    % for p=1:50
    %     eigen_face_1(p,:) = E(1,(50*(p-1)+1):50*p);
    % end
    % figure;
    % imshow(eigen_face_1);

    %Finding Projections
    X_database = transpose(V)*X_dev;
%end
end