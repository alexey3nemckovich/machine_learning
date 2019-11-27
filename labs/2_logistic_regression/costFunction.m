function [J, grad] = costFunction(theta, X, y)
  
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

coef = 1 / m;

% Calculation J

temp_sum = sum(-y .* log(sigmoid(X * theta)) .- ((1 - y) .* log(1 - sigmoid(X * theta))));
J =  coef * (temp_sum);

% Calculation gradient

grad_size = length(grad);
for iter_grad = 1:grad_size
	grad(iter_grad) = coef * sum(( sigmoid(X * theta) .- y) .* X(:,iter_grad));
end

% =============================================================

end
