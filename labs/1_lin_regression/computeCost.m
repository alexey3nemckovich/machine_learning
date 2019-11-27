function J = computeCost(X, y, theta)

m = length(y); % number of training examples

J = 0;

coef = 1 / (2 * m);
square_sum = sum((X * theta - y) .^ 2);
J =  coef * (square_sum);

end

