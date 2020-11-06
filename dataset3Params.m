function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C=[0.01 0.03 0.1 0.3 1 3 10 30];
sigma=[0.01 0.03 0.1 0.3 1 3 10 30];
Values=zeros(64,3);
d=1;

for b=1:size(sigma,2);
sigma_val=sigma(1,b);
for a=1:size(C,2);
C_val=C(1,a);
model = svmTrain(X, y, C_val, @(x1, x2)gaussianKernel(x1, x2, sigma_val));
predictions = svmPredict(model, Xval);
error=mean(double(predictions ~= yval));
Values(d,1)=C_val;
Values(d,2)=sigma_val;
Values(d,3)=error;
d=d+1;
end
end
[M,I] = min(Values(:,3));

C=Values(I,1);
sigma=Values(I,2);

% =========================================================================

end
