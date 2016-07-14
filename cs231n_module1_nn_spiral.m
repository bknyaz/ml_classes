% Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition. 
% Module 1
% Matlab version of scripts from http://cs231n.github.io/neural-networks-case-study/

%% Generating some data
N = 100;
D = 2;
K = 3;
X = zeros(N*K,D);
y = zeros(N*K,1);
for j=1:K
  ix = N*(j-1)+1:N*j;
  r = linspace(0,1,N); % radius
  t = linspace((j-1)*4+1,j*4,N)+randn(1,N).*0.2; % theta
  X(ix,:) = [r.*sin(t);r.*cos(t)]';
  y(ix) = j;
end
close all
figure, scatter(X(:,1), X(:,2), 40, y, 'filled', 'MarkerEdgeColor', 'black'), colormap jet
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
for i=0:200
  
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

%% visualize decision boundaries
[w1,w2] = meshgrid(-1.5:0.01:1.5,-1.5:0.01:1.5);
scores = bsxfun(@plus, [w1(:),w2(:)] * W, b);
exp_scores = exp(scores);
probs = bsxfun(@times, exp_scores, 1./sum(exp_scores,2));
figure, imagesc(w1(:), w2(:), reshape(probs(:,[3,2,1]),[size(w1),3])), colormap jet
hold on, scatter(X(:,1), X(:,2), 40, y, 'filled', 'MarkerEdgeColor', 'black')
title('The toy spiral data with three classes: Linear Classifier')

%% Training a Neural Network
% initialize parameters randomly
h = 100; % size of hidden layer
W = 0.01 .* randn(D,h);
b = zeros(1,h);
W2 = 0.01 .* randn(h,K);
b2 = zeros(1,K);

for i=0:10000
  
  % evaluate class scores, [N x K]
  % evaluate class scores with a 2-layer Neural Network
  hidden_layer = max(0, bsxfun(@plus, X * W, b)); % note, ReLU activation
  scores = bsxfun(@plus, hidden_layer * W2, b2);

  % compute the class probabilities
  exp_scores = exp(scores);
  probs = bsxfun(@times, exp_scores, 1./sum(exp_scores,2)); % [N x K]
  
  % compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -log(probs(sub2ind(size(probs),[1:num_examples]',y)));
  data_loss = sum(corect_logprobs)/num_examples;
  reg_loss = 0.5.*reg*sum(sum(W.*W)) + 0.5.*reg*sum(sum(W2.*W2));
  loss = data_loss + reg_loss;
  if mod(i,1000) == 0
    fprintf('iteration %d: loss %f \n', i, loss)
  end
  
  % compute the gradient on scores
  dscores = probs;
  ids = sub2ind(size(probs),[1:num_examples]',y);
  dscores(ids) = dscores(ids) - 1;
  dscores = dscores./num_examples;
  
  % backpropate the gradient to the parameters
  % first backprop into parameters W2 and b2
  dW2 = hidden_layer' * dscores;
  db2 = sum(dscores, 1);
  
  % next backprop into hidden layer
  dhidden = dscores * W2';
  % backprop the ReLU non-linearity
  dhidden(hidden_layer <= 0) = 0;
  % finally into W,b
  dW = X' * dhidden;
  db = sum(dhidden, 1);
  
  % add regularization gradient contribution
  dW2 = dW2 + reg .* W2;
  dW = dW + reg .* W;
  
  % perform a parameter update
  W = W - step_size .* dW;
  b = b - step_size .* db;
  W2 = W2 - step_size .* dW2;
  b2 = b2 - step_size .* db2;
end
% evaluate training set accuracy
hidden_layer = max(0, bsxfun(@plus, X * W, b));
scores = bsxfun(@plus, hidden_layer * W2, b2);
[~,predicted_class] = max(scores, [], 2);
fprintf('training accuracy: %.2f \n', mean(predicted_class == y))

%% visualize decision boundaries
[w1,w2] = meshgrid(-1.5:0.01:1.5,-1.5:0.01:1.5);
hidden_layer = max(0, bsxfun(@plus, [w1(:),w2(:)] * W, b));
scores = bsxfun(@plus, hidden_layer * W2, b2);
exp_scores = exp(scores);
probs = bsxfun(@times, exp_scores, 1./sum(exp_scores,2));
figure, imagesc(w1(:), w2(:), reshape(probs(:,[3,2,1]),[size(w1),3])), colormap jet
hold on, scatter(X(:,1),X(:,2), 40, y, 'filled', 'MarkerEdgeColor', 'black')
title('The toy spiral data with three classes: Neural Network')