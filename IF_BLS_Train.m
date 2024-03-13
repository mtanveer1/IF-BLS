function [train_accuracy,Model,train_time] = IF_BLS_Train(dataTrain,S,option)

train_x=dataTrain(:,1:end-1);
trainY=dataTrain(:,end);

[Nsample,~]=size(train_x);
dataY_train=trainY;

U_dataY_train = [0,1];
nclass=2; %Total number of class (Classification Problem)
dataY_train_temp = zeros(numel(dataY_train),nclass); 

% 0-1 coding for the target
for i=1:nclass
    idx = dataY_train==U_dataY_train(i);
    dataY_train_temp(idx,i)=1;
end

N1=option.n1;
N2=option.n2;
N3=option.n3;
N4=1;
C=option.c;
mew=option.mu;

tic
H1 = [train_x .1 * ones(size(train_x,1),1)];
Z=[];
for i=1:N2
    we=2*rand(size(train_x,2)+1,N1)-1;
    We{i}=we;
    A1 = H1 * we;
    A1 = mapminmax(A1);
    Z=[Z,A1];
    clear we;
    clear A1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Enhancement Layer%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:N4 %This loop is not taken in the original code, they just took only one group in the enhancement layer
H2 = [Z .1 * ones(size(trainY,1),1)];
if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
end
Wh{i}=wh;
H=[];
   
    A2 = H2 * wh;
    A2=tansig(A2);
    H=[H,A2];
    clear wh;
end

A=[Z,H];

%%%%%%%%%%%%%%%% IFBLS SOLUTION %%%%%%%%%%%%%%
if size(A,2)<Nsample %i.e. No of columns of X is less than Nsample
    W = (eye(size(S*A,2))/C+(S*A)'*(S*A)) \ (S*A)'*S*dataY_train_temp; 
else
    lambda=0.0001;
     W = A'*((inv(S+(lambda*(eye(size(A,1)))))*inv(S+(lambda*(eye(size(A,1)))))/C+A*A') \ dataY_train_temp);
end
Model.W=W;
Model.Wh=Wh;
Model.We=We;
trainY_temp=A*W;

train_time=toc;

%%%%%%%%%%%%%%%%% TRAINING ACCURACY FOR CLASSIFICATION PROBLEM
%softmax to generate probabilites
trainY_temp1 = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
num = exp(trainY_temp1);
dem = sum(num,2);
prob_scores = bsxfun(@rdivide,num,dem);
[~,indx] = max(prob_scores,[],2);
[~, ind_corrClass] = max(dataY_train_temp,[],2);
train_accuracy = mean(indx == ind_corrClass)*100;
end