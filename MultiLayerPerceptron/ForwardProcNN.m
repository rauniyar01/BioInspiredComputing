%Author: Kiarie Ndegwa u4742829
%Assignment 1 comp8420: Multi-layer and cascade neural networks for
%the classification of Diabete Mellitus.

%This project works on classifying different liver disorders based
%on a small data set comprised of 768 entries

%The paper accompanying the data set achieved a classification accuracy 76%.
clear
clc
%Keep the random seed the same during all network initialization during
%testing
rng('default')
DiabetesData = csvread('IndianDiabetesData.csv');

%Divide the data set into input attributes and binary outputs
inputs = DiabetesData(:, 1:end-1)';
outputs = DiabetesData(:, end)';

%Data pre-processing
%-----------------------------------------------------------------%
%STEP 1: PCA dimension reduction

%Step skipped as PCA doesn't reduce the dimensionality of the input
%PCA = pca(inputs);

%STEP 2: Scale inputs between -1 and 1
ProcInputs = bsxfun(@minus, inputs, mean(inputs));
%STEP 3: Normalize the PCA components.
ProcInputs = normc(ProcInputs);
%----------------------------------------------------------------%

%Create validation indices with 10 fold cross validation
%----------------------------------------------------------------%
CVO = cvpartition(outputs(1, :), 'k', 10);

%These vectors keeps record of the test error on unprocessed data
errForwardUn = zeros(CVO.NumTestSets,1);
%---------------------------------------------------------------%

%---------------------------------------------------------------%
%------This part of the code uses unprocessed input data--------%
%---------------------------------------------------------------%

%This vector keeps record of the test error
errForwardProc = zeros(CVO.NumTestSets,1);
errCascProc = zeros(CVO.NumTestSets,1);

%This tests and trains the multi-layer network
for i = 1:CVO.NumTestSets
     
     trIdxProc = CVO.training(i);
     teIdxProc = CVO.test(i);
     train_inputsProc= ProcInputs(:,trIdxProc);
     train_outputsProc= outputs(:,trIdxProc);
     test_inputsProc= ProcInputs(:,teIdxProc);
     test_outputsProc= outputs(:,teIdxProc);
    
     netFowardProc = feedforwardnet([4, 2]);
     netFowardProc = train(netFowardProc, train_inputsProc, train_outputsProc);
    
     %Find test error on both architectures.
     y_netForwardProc = netFowardProc(test_inputsProc);
     
     errForwardProc(i) = sum(round(y_netForwardProc)~=test_outputsProc)/length(test_outputsProc);
     
end

%Find accuracy of trained networks on unprocessed data
errForwardProc = sum(errForwardProc)/CVO.NumTestSets;
X = sprintf('Classification accuracy on unprocessed input data for multilayered network %.2f%%',(1-errForwardProc)*100);
disp(X);