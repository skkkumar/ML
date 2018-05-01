% mnist from original file
load mnist_all.mat
mnist_images = double([train0; train1; train2; train3; train4; train5; train6; train7; train8; train9]');
% output with one-hot coding
temp = eye(10);
data_output = [repmat(temp(:,1),[1 size(train0,1)]) repmat(temp(:,2),[1 size(train1,1)]) ...
               repmat(temp(:,3),[1 size(train2,1)]) repmat(temp(:,4),[1 size(train3,1)]) ...
               repmat(temp(:,5),[1 size(train4,1)]) repmat(temp(:,6),[1 size(train5,1)]) ...
               repmat(temp(:,7),[1 size(train6,1)]) repmat(temp(:,8),[1 size(train7,1)]) ...
               repmat(temp(:,9),[1 size(train8,1)]) repmat(temp(:,10),[1 size(train9,1)])];
clear train0 train1 train2 train3 train4 train5 train6 train7 train8 train9
data_input = mnist_images;
scale = max(data_input(:)) - min(data_input(:));
data_input = data_input / scale;
data_mean = mean(data_input, 2);
data_input = bsxfun(@minus, data_input, data_mean);
clear mnist_images
% shuffle training examples
I = randperm(size(data_output,2));
data_input = data_input(:, I);
data_output = data_output(:, I);
% validation data
mnist_valid_images = double([test0; test1; test2; test3; test4; test5; test6; test7; test8; test9]');
valid_input = mnist_valid_images / scale;
valid_input = bsxfun(@minus, valid_input, data_mean);
valid_output =[repmat(temp(:,1),[1 size(test0,1)]) repmat(temp(:,2),[1 size(test1,1)]) ...
               repmat(temp(:,3),[1 size(test2,1)]) repmat(temp(:,4),[1 size(test3,1)]) ...
               repmat(temp(:,5),[1 size(test4,1)]) repmat(temp(:,6),[1 size(test5,1)]) ...
               repmat(temp(:,7),[1 size(test6,1)]) repmat(temp(:,8),[1 size(test7,1)]) ...
               repmat(temp(:,9),[1 size(test8,1)]) repmat(temp(:,10),[1 size(test9,1)])];
% save preprocessed file
save('mnist_preprocessed.mat', ...
     'data_input','data_output','valid_input','valid_output');

% show examples
figure(8);
mnist_images = mnist_images(:, I);
visualize(mnist_images(:,1:20));