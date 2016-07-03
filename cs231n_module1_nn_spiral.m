% Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition. 
% Module 1
% http://cs231n.github.io/neural-networks-case-study/

%% Generating some data
N = 100;
D = 2;
K = 3;
X = zeros(N*K,D);
y = zeros(N*K,1);
for j=1:K
    ix = N*(j-1)+1:N*j;
    r = linspace(0,1,N);
    t = linspace((j-1)*4+1,j*4,N)+randn(1,N).*0.2;
    X(ix,:) = [r.*sin(t);r.*cos(t)]';
    y(ix) = j;
end
scatter(X(:,1),X(:,2),4,...
    cat(1,bsxfun(@plus,zeros(N,3),[1,0,0]),...
    bsxfun(@plus,zeros(N,3),[0.5,0.5,0]),...
    bsxfun(@plus,zeros(N,3),[0,0,1])))
title('The toy spiral data with three classes')

%% Initialize the parameters
W = 0.01 .* randn(D,K);
b = zeros(1,K);

%% Compute the class scores
scores = bsxfun(@plus, X * W, b);

%% Compute the loss
% get unnormalized probabilities
exp_scores = exp(scores);
% normalize them for each example
num_examples = size(X,1);
probs = bsxfun(@times, exp_scores, 1./sum(exp_scores,2));
corect_logprobs = -log(probs(sub2ind(size(probs),[1:num_examples]',y)));

% compute the loss: average cross-entropy loss and regularization

% some hyperparameters
step_size = 1e-0;
reg = 1e-3; % regularization strength (lambda)

data_loss = sum(corect_logprobs)/num_examples;
reg_loss = 0.5.*reg*sum(sum(W.*W));
loss = data_loss + reg_loss

%% Computing the Analytic Gradient with Backpropagation
dscores = probs;
ids = sub2ind(size(probs),[1:num_examples]',y);
dscores(ids) = dscores(ids) - 1;
dscores = dscores./num_examples;

dW = X'*dscores;
db = sum(dscores,1);
dW = dW + reg*W; % don't forget the regularization gradient

%% Performing a parameter update
% perform a parameter update
W = W - step_size * dW;
b = b - step_size * db;

%% Putting it all together: Training a Softmax Classifier
% Train a Linear Classifier

% initialize parameters randomly
W = 0.01 .* randn(D,K);
b = zeros(1,K);

% gradient descent loop

for i=1:200
  
  % evaluate class scores, [N x K]
  scores = bsxfun(@plus, X * W, b);
  
  % compute the class probabilities
  exp_scores = exp(scores);
  probs = bsxfun(@times, exp_scores, 1./sum(exp_scores,2));
  
  
  % compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -log(probs(sub2ind(size(probs),[1:num_examples]',y)));
  data_loss = sum(corect_logprobs)/num_examples;
  reg_loss = 0.5.*reg*sum(sum(W.*W));
  loss = data_loss + reg_loss;
  if mod(i,10) == 0
    fprintf('iteration %d: loss %f \n', i, loss)
  end
  
  % compute the gradient on scores
  dscores = probs;
  ids = sub2ind(size(probs),[1:num_examples]',y);
  dscores(ids) = dscores(ids) - 1;
  dscores = dscores./num_examples;

  % backpropate the gradient to the parameters (W,b)
  dW = X'*dscores;
  db = sum(dscores,1);

  dW = dW + reg*W; % don't forget the regularization gradient

  % perform a parameter update
  W = W - step_size * dW;
  b = b - step_size * db;
 
end

% evaluate training set accuracy
scores = bsxfun(@plus, X * W, b);
[~,predicted_class] = max(scores, [], 2);
fprintf('training accuracy: %.2f \n', mean(predicted_class == y))