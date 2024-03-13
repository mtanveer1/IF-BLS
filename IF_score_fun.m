function [S]=IF_score_fun(A,mew)
%function [S1]=score_fun(K1,K2,K3,no_input,label)
%K1 is the kernel matrix corresponding to positive class data point and K2
%is kenel matrix corresponding to negative class data point.
[no_input,no_col]=size(A);
A1=A(A(:,end)==1,1:end-1);
B1=A(A(:,end)~=1,1:end-1);
label=A(:,end);
K1 = exp(-(1/(mew^2))*(repmat(sqrt(sum(A1.^2,2).^2),1,size(A1,1))-2*(A1*A1')+repmat(sqrt(sum(A1.^2,2)'.^2),size(A1,1),1)));
K2 = exp(-(1/(mew^2))*(repmat(sqrt(sum(B1.^2,2).^2),1,size(B1,1))-2*(B1*B1')+repmat(sqrt(sum(B1.^2,2)'.^2),size(B1,1),1)));
A_temp=A(:,1:end-1);
K3 = exp(-(1/(mew^2))*(repmat(sqrt(sum(A_temp.^2,2).^2),1,size(A_temp,1))-2*(A_temp*A_temp')+repmat(sqrt(sum(A_temp.^2,2)'.^2),size(A_temp,1),1)));

radiusxp=sqrt(1-2*mean(K1,2)+mean(mean(K1)));
radiusmaxxp=max(radiusxp);
radiusxn=sqrt(1-2*mean(K2,2)+mean(mean(K2)));
radiusmaxxn=max(radiusxn);

alpha_d=max(radiusmaxxn,radiusmaxxp); 

mem=[];
j=1;k=1;
for i=1:no_input
    if(label(i)==1)
        membership=1-(radiusxp(j)/(radiusmaxxp+10^-4));
        j=j+1;
    else
        membership=1-(radiusxn(k)/(radiusmaxxn+10^-4));
        k=k+1;
    end
    mem=[mem membership];
end
    
    % membership1=ones(size(radiusxp,1),1)-(radiusxp/(radiusmaxxp+10^-4));
    % membership2=ones(size(radiusxn,1),1)-(radiusxn/(radiusmaxxn+10^-4));
    % mem=[membership1; membership2];
    %end
    
    ro=[];
    DD=sqrt(2*(ones(size(K3))-K3));
    DD=real(DD);
    for i=1:no_input
        
        temp=DD(i,:)';
        B1=A(temp<alpha_d,:);
        
        [x3,~]=size(B1);
        count=sum(A(i,end)*ones(size(B1,1),1)~=B1(:,end));
        
        x5=count/x3;
        ro=[ro;x5];
    end
    
    %A2=[A(:,no_col) ro];
    %ro2=A2(A2(:,1)==-1,2);
    %ro1=A2(A2(:,1)~=-1,2);
    
    v=(ones(size(mem))-mem).*ro;
    
    S=[];
    for i=1:size(v,1)
        if v(i)==0
            S=[S;mem(i)];
        elseif (mem(i)<=v(i))
            S=[S;0];
        else
            S=[S;(1-v(i))/(2-mem(i)-v(i))];
        end
    end
%     S=diag(S);
end