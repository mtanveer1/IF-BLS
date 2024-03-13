% Please cite the following paper if you are using this code.
% Reference: M. Sajid, A. K. Malik, and M. Tanveer. "Intuitionistic Fuzzy Broad Learning System: Enhancing Robustness Against Noise and Outliers." arXiv preprint arXiv:2307.08713 (2023).
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The experimental procedures are executed on a computing system possessing MATLAB R2023a software, an 11th Gen Intel(R) Core(TM) i7-11700 processor operating at 2.50GHz with 16.0 GB RAM, 
% and a Windows-11 operating platform.
% 
% We have put a demo of the "Intuitionistic Fuzzy BLS" model with the "credit_approval" dataset 

%%
clc;
clear;
warning off all;
format compact;

%% Data Preparation
split_ratio=0.8; nFolds=5; addpath(genpath('C:\Users\HP\OneDrive - IIT Indore\Desktop\IF-BLS\Github Code'))
temp_data1=load('credit_approval.mat');

temp_data=temp_data1.credit_approval;

[Cls,~,~] = unique(temp_data(:,end));
No_of_class = size(Cls,1);


trainX=temp_data(:,1:end-1); mean_X = mean(trainX,1); std_X = std(trainX);
trainX = bsxfun(@rdivide,trainX-repmat(mean_X,size(trainX,1),1),std_X);
All_Data=[trainX,temp_data(:,end)];

[samples,~]=size(All_Data);
% rng('default') % Fix seed value if user wants, otherwise take mean value
% of all the folds for average accuracy
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
test_x=testing_Data(:,1:end-1); test_y=testing_Data(:,end);
train_x=training_Data(:,1:end-1); train_y=training_Data(:,end);
Train_Data=[train_x,train_y]; Test_Data=[test_x,test_y];

%% Hyperparameter range
%C = [10^{-6}, 10^{-4}, ..., 10^{6}]
%m = 1 : 2 : 21
%p = 5 : 5 : 50
%q = 5 : 10 : 105
%mu = [2^{−5}, 2^{−4}, · · · , 2^5]

%%
C=100; %Regularization parameter
mu=4; %Kernel parameter
p=10; %Number of fuzzy nodes in each groups
m=1; %Number of fuzzy groups
q=85; %Number of enhancement nodes

option.c=C; option.mu=mu; option.n1=p; option.n2=m; option.n3=q;
%% Intuitionistic Fuzzy Score
Score=IF_score_fun(Train_Data,mu);
S=diag(Score);

%% Calling training function
[TrainingAccuracy,TestingAccuracy,TrainingTime,TestingTime] = IF_BLS_Classification(Train_Data,Test_Data,S,option);

%% This accuracy is with respect to one fold. Take mean for average accuracy and average standard deviation
fprintf(1, 'Testing Accuracy of IF-BLS model is: %f\n', TestingAccuracy); 

