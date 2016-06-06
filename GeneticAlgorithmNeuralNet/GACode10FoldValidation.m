%Assignment 2:
%Genetic algorithms for neural network weight selection
clear
clc
%Use this function to optimize the weights of the best net generated
%in the previous assignment.

%Keep the random seed the same during all network initialization during
%testing
rng('default');

PimaDataSet = csvread('IndianDiabetesData.csv');
%Training data
inputs = PimaDataSet(:, 1:end-1)';
targets = PimaDataSet(:, end)';

%----------------------------------------------------------------%
%Create validation indices with 10 fold cross validation
%----------------------------------------------------------------%
CVO = cvpartition(targets(1, :), 'k', 10);

%These vectors keeps record of the test error on unprocessed data
err = zeros(CVO.NumTestSets,1);
for i = 1:CVO.NumTestSets
         trIdxUn = CVO.training(i);
         teIdxUn = CVO.test(i);
         train_inputs = inputs(:,trIdxUn);
         train_targets = targets(:,trIdxUn);
         test_inputs = inputs(:,teIdxUn);
         test_targets= targets(:,teIdxUn);

        %INITIALIZE THE NEURAL NETWORK PROBLEM %
        % number of neurons
        Layer1 = 4;
        Layer2 = 3;

        % create a neural network
        net = feedforwardnet([Layer1,Layer2]);
        % configure the neural network for this dataset
        net = configure(net, train_inputs, train_targets);
        % create handle to the MSE_TEST function, that
        % calculates MSE/ this acts as the fitness/objective function
        h = @(x) mse_test(x, net, train_inputs, train_targets);
        % Setting the Genetic Algorithms tolerance for
        % minimum change in fitness function before
        % terminating algorithm to 1e-3 and displaying
        % each iteration's results.
        ga_opts = gaoptimset('TolFun', 1e-3,'display','iter');

        %Get number of weights that need to be learned/optimized by GA
        no_weights = size(getwb(net), 1);
        [x_ga_opt, err_ga] = ga(h, no_weights, ga_opts);

        %Find accuracy of weight trained network
        %Set up new network with trained weights
         net = setwb(net, x_ga_opt');
         %Make sure all outputs are binary integers by rounding to nearest integer
         GA_out = round(net(test_inputs));
         %Find accuracy of generated figure against targer values given new data
         err(i) = sum(GA_out~=test_targets)/length(test_targets);
end

%Find accuracy of trained networks on unprocessed data
err = sum(err)/CVO.NumTestSets;

X = sprintf('Classification accuracy on GA MLP network %.2f%%',(1-err)*100);
disp(X);