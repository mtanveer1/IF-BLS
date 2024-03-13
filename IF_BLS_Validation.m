function [validation_accuracy,valid_time] = IF_BLS_Validation(dataTest,Model,option)
nclass=2;
test_x=dataTest(:,1:end-1);
testY=dataTest(:,end);

%%%%%%%%%%%%% TEST DATA %%%%%%%%%%%%%%
dataY_test=testY;
U_dataY_test = [0,1];
nclass=2;
dataY_test_temp = zeros(numel(dataY_test),nclass);

% 0-1 coding for the target
for i=1:nclass 
    idy = dataY_test==U_dataY_test(i);
    dataY_test_temp(idy,i)=1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N1=option.n1;
N2=option.n2;
N3=option.n3;
N4=1;
C=option.c;
mew=option.mu;

W=Model.W;
Wh=Model.Wh;
We=Model.We;

Z_test=[];

tic
T1 = [test_x .1 * ones(size(test_x,1),1)];

for i=1:N2
    T2 = T1 * We{i};
    T2 = mapminmax(T2);
    Z_test=[Z_test,T2];
end

for i=1:N4 %This loop is not taken in the original code, they just took only one group in the enhancement layer
I2 = [Z_test .1 * ones(size(testY,1),1)];
H_test=[];
    S2 = I2 * Wh{i};
    S2=tansig(S2);
    H_test=[H_test,S2];
    clear S2;
end

B=[Z_test,H_test];

testY_temp=B*W;
valid_time=toc;

%%%%%%%%%%%%%%%%% TESTING ACCURACY FOR CLASSIFICATION PROBLEM
%softmax to generate probabilites
testY_temp1 = bsxfun(@minus,testY_temp,max(testY_temp,[],2)); %for numerical stability
num = exp(testY_temp1);
dem = sum(num,2);
prob_scores = bsxfun(@rdivide,num,dem);
[~,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(dataY_test_temp,[],2);
validation_accuracy = mean(indx == ind_corrClass)*100;
end
%EOF