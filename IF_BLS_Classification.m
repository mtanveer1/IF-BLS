function [train_accuracy,validation_accuracy,train_time,valid_time] = IF_BLS_Classification(dataTrain,dataTest,S,option)

%seed = RandStream('mcg16807','Seed',0);
%RandStream.setGlobalStream(seed);

% Train RVFL
[train_accuracy,Model,train_time] = IF_BLS_Train(dataTrain,S,option);

% Using trained model, predict the testing data
[validation_accuracy,valid_time] = IF_BLS_Validation(dataTest,Model,option);

end
%EOF