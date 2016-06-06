%Assignment 2:
%Genetic algorithms for neural network weight selection
clear
clc
%Use this function to optimize the weights of the best net generated
%in the previous assignment.

%Keep the random seed the same during all network initialization during
%testing

PimaDataSet = csvread('IndianDiabetesData.csv');
%Training data
inputs = PimaDataSet(1:384, 1:end-1)';
targets = PimaDataSet(1:384, end)';
%Test data
test_inputs = PimaDataSet(385:end, 1:end-1)';
test_targets = PimaDataSet(385:end, end)';

%INITIALIZE THE NEURAL NETWORK PROBLEM %
% number of neurons
Layer1 = 4;
Layer2 = 2;

% create a neural network
net = cascadeforwardnet([Layer1,Layer2]);
% configure the neural network for this dataset
net = configure(net, inputs, targets);
% create handle to the MSE_TEST function, that
% calculates MSE/ this acts as the fitness/objective function
h = @(x) mse_test(x, net, inputs, targets);
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
 error = sum(GA_out~=test_targets)/length(test_targets);
 