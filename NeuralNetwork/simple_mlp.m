load mnist_preprocessed.mat
layers = [784 225 144 10];
nlayers = length(layers)-1;
minibatchsize = 100;
niters = 60000/minibatchsize * 50; % 50 epochs
momentum = 0.9;
regularization_noise = 0.4;
origstepsize = 1.0;
[ndim,nsamples] = size(data_input);
[nclasses,nsamples] = size(data_output);
% initialization of weights and biases
for l = 1:nlayers,
 W{l} = sqrt(2)/sqrt(layers(l+1)+layers(l))*randn(layers(l+1), layers(l));
 b{l} = 0.1*ones(layers(l+1), 1);
end

% initialize momentum and transformation vectors to 0
for l=1:nlayers, 
  dirb{l} = 0; 
  dirW{l} = 0;
end
% training:
for iter = 1 : niters,
 stepsize = origstepsize * min([1; 10*iter/niters; 2*(niters+1-iter)/niters]);
 % pick the next minibatch
 input = data_input(1:ndim, mod((iter-1)*minibatchsize, nsamples)+1 : mod(iter*minibatchsize-1, nsamples)+1);
 input = input + regularization_noise * randn(size(input));
 output = data_output(1:nclasses, mod((iter-1)*minibatchsize, nsamples)+1 : mod(iter*minibatchsize-1, nsamples)+1);
 clear a h softmax
 h{1} = input;
 % feedforward loop
 for l = 1:nlayers,
  a{l+1} = repmat(b{l}, 1, minibatchsize) + W{l} * h{l};
  h{l+1} = max(a{l+1}, 0);
 end 
 softmax = exp(a{nlayers+1}) ./ repmat(sum(exp(a{nlayers+1}), 1), nclasses, 1);
 loss = -log(sum( softmax .* output, 1));
 % compute the  gradient
  for l = nlayers:-1:1,
   if (l==nlayers)
    grad{nlayers} = (softmax - output);
   else
     % back-prop
     grad{l} = (a{l+1} > 0) .* (W{l+1}' * grad{l+1});
   end
   grad_b{l} = mean(grad{l}, 2);
   grad_W{l} = grad{l} * h{l}' / minibatchsize;
  end
 % do the actual updates
 for l = 1:nlayers,
  dirb{l} = momentum * dirb{l} + (1 - momentum) * grad_b{l};
  dirW{l} = momentum * dirW{l} + (1 - momentum) * grad_W{l};
  b{l} = b{l} - stepsize * dirb{l};
  W{l} = W{l} - stepsize * dirW{l};
 end
 % things to plot after every epoch
 curves(iter, 1) = mean(mean(loss));
 if (mod(iter, size(data_input, 2) / minibatchsize) == 0),
  epoch = ceil(iter/(size(data_input, 2) / minibatchsize));
  windowedloss = mean(curves((iter-size(data_input, 2) / minibatchsize+1):iter, 1));
  res.loss(epoch) = windowedloss;
  clear a h softmax
  for traintest = 1:2,
   if (traintest==1)
     h = data_input;
   else
     h = valid_input;
   end
   for l = 1:nlayers,
       a = repmat(b{l}, 1, size(h, 2)) + W{l} * h;
       h = max(a,0);
   end
   softmax{traintest} = exp(a) ./ repmat(sum(exp(a), 1), nclasses, 1);
  end
  train_acc = mean(min(data_output == (softmax{1} == repmat(max(softmax{1}, [], 1), nclasses, 1)), [], 1));
  accuracy = mean(min(valid_output == (softmax{2} == repmat(max(softmax{2}, [], 1), nclasses, 1)), [], 1));
  res.train(epoch) = train_acc;
  res.test(epoch) = accuracy;
  res.stepsize(epoch) = stepsize;
  fprintf('epoch %d: stepsize %4.2f, train accu %6.4f, test accu %6.4f\n', epoch, stepsize, train_acc, accuracy);
  % show first layer weights after every epoch:
  figure(1);
  visualize(W{1}');
%  print(sprintf('weights_%2d',epoch),'-dpng');
 end
end

% ***********************************************
% end of training, the rest is for making figures
% ***********************************************

% show learning curves
figure(2);
plot([100*(1-res.train);100*(1-res.test)]');
legend('training error %','test error %');
title('Learning curves');
axis([1 length(res.train) 0 5]);

% do feedforward on validation data for histograms
clear a h
h{1} = valid_input;
for l = 1:nlayers,
  a{l+1} = repmat(b{l}, 1, size(h{l}, 2)) + W{l} * h{l};
  h{l+1} = max(a{l+1},0);
end
figure(3);
hist(sum(double(h{2}>0),2)/size(h{2},2),50)
figure(13);
hist(sum(double(h{3}>0),2)/size(h{3},2),50)

% do feedforward on couple of examples to plot activations
clear a h
h{1} = [valid_input(:,500:1000:end) data_input(:,1:6)+regularization_noise*randn(ndim,6)];
for l = 1:nlayers,
  a{l+1} = repmat(b{l}, 1, size(h{l}, 2)) + W{l} * h{l};
  h{l+1} = max(a{l+1},0);
end
figure(4);
visualize(h{1})
title('inputs');
figure(5);
visualize(h{2})
title('hidden activations 1');
figure(6);
visualize(h{3});
title('hidden activations 2');
figure(7);
visualize([exp(a{4}) ./ repmat(sum(exp(a{4}), 1), nclasses, 1); zeros(6,16)])
title('class probabilities');

% show weights
figure(11);
visualize(W{2}');
figure(12);
visualize(W{3}');

