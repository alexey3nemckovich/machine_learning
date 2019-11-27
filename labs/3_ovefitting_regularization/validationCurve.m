function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
	train_theta = trainLinearReg(X, y, lambda_vec(i));
	error_train(i) = (1 / (2 * size(X,1)) ) * sum((X * train_theta .- y) .^ 2);
	error_val(i) = (1 / (2 * size(Xval,1)) ) * sum((Xval * train_theta .- yval) .^ 2);
end


% =========================================================================

end
