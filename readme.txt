Please cite the following paper if you are using this code.

Reference: M. Sajid, A. K. Malik, and M. Tanveer. "Intuitionistic Fuzzy Broad Learning System: Enhancing Robustness Against Noise and Outliers." arXiv preprint arXiv:2307.08713 (2023).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The experimental procedures are executed on a computing system possessing MATLAB R2023a software, an 11th Gen Intel(R) Core(TM) i7-11700 processor operating at 2.50GHz with 16.0 GB RAM, and a Windows-11 operating platform.

We have put a demo of the "Intuitionistic Fuzzy BLS" model with the "credit_approval" dataset 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Hyperparameter range
--------------------
%C = [10^{-6}, 10^{-4}, ..., 10^{6}]
%m = 1 : 2 : 21
%p = 5 : 5 : 50
%q = 5 : 10 : 105
%mu = [2^{−5}, 2^{−4}, · · · , 2^5]

The following are the parameters set used for the experiment 
------------------------------------------------------------
C=100; %Regularization parameter
mu=4; %Kernel parameter
p=10; %Number of fuzzy nodes in each groups
m=1; %Number of fuzzy groups
q=85; %Number of enhancement nodes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Description of files:
---------------------
IF_BLS_Main.m: This is the main file to run selected models on datasets. In the path variable specificy the path to the folder containing the codes and datasets on which you wish to run the algorithm. 

IF_BLS_Classification: This is an intermediate function, from where train and validation functions are called.

IF_BLS_Train.m: This is the training function of the model.

IF_BLS_Validation.m: This is the validation function of the model.

IF_score_fun.m: Intuitionistic fuzzy score function.

credit_approval.mat: Dataset

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The codes are not optimized for efficiency. The codes have been cleaned for better readability and documented and are not exactly the same as used in our paper.
For the detailed experimental setup, please follow the paper. 
We have re-run and checked the codes only in a few datasets, so if you find any bugs/issues, please write to M. Sajid (phd2101241003@iiti.ac.in).


Some parts of the codes have been taken from:
1. CL Philip Chen, and Zhulin Liu. "Broad learning system: An effective and efficient incremental learning system without the need for deep architecture." IEEE transactions on neural networks and learning systems 29, no. 1 (2017): 10-24.

13-March-2024

