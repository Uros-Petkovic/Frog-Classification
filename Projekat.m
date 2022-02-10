
%% Ucitavanje podataka i podela na trening i test skup

clc; clear all;

Podaci=readtable('Frogs_MFCCs.csv');   %Ucitavanje fajla
Podaci1=Podaci(:,1:22);      %Prve 22 kolone su obelezja po 7195 odbiraka
Podaci1=table2array(Podaci1);%Vrsimo prebacivanje iz table u double
D=[Podaci1(673:1214,:)];     %Odvajamo podatke za klasu D
B=[Podaci1(6980:7047,:)];    %Odvajamo podatke za klasu B,L i H
L=[Podaci1(1:672,:); Podaci1(1215:4692,:); Podaci1(6596:6865,:)];
H=[Podaci1(4693:6595,:); Podaci1(6866:6979,:);Podaci1(7048:7195,:) ];

XL=L;  %Smestamo ih u ove vektore
XD=D;
XH=H;
XB=B;

%Nalazimo broj elemenata u svakoj klasi i mesamo ih radi boljeg obucavanja
NL=length(XL); indL=randperm(NL); XL=XL(indL,:);
ND=length(XD); indD=randperm(ND); XD=XD(indD,:);
NH=length(XH); indH=randperm(NH); XH=XH(indH,:);
NB=length(XB); %Ovu klasu ne koristimo

%Prikaz klasa u zavisnosti od broja podataka
figure(1); stem(1,NL); hold on;
stem(2,ND); hold on;
stem(3,NH); hold on;
stem(4,NB); hold off;
title('Klase u zavisnosti od broja odbiraka');
xlabel('Klasa');
ylabel('Broj odbiraka');
legend('Leptodactylidae','Dendrobatidae','Hylidae','Bufonidae');
xlim([-1 5]);

%Uzimamo 70% podataka za trening i 30% za test
%Zbog malog broja podataka klase 4,koristimo samo 3 klase pri cemu izlaz
%svake klase predstavlja trobitni broj 001,010 ili 100

XLtrening=XL(1:round(0.7*NL),:);  %Trening
YLtrening=zeros(round(0.7*NL),3); %Trening izlaz 001
YLtrening(:,3)=1;
XLtest=XL(round(0.7*NL)+1:end,:); %Test
YLtest=zeros(round(0.3*NL),3);    %Test izlaz 001
YLtest(:,3)=1;

XDtrening=XD(1:round(0.7*ND),:);  %Trening
YDtrening=zeros(round(0.7*ND),3); %Trening izlaz 010
YDtrening(:,2)=1;
XDtest=XD(round(0.7*ND)+1:end,:); %Test
YDtest=zeros(round(0.3*ND),3);    %Test izlaz 010
YDtest(:,2)=1; 


XHtrening=XH(1:round(0.7*NH),:);  %Trening
YHtrening=zeros(round(0.7*NH),3); %Trening izlaz 100
YHtrening(:,1)=1;
XHtest=XH(round(0.7*NH)+1:end,:); %Test
YHtest=zeros(round(0.3*NH),3);    %Test izlaz 100
YHtest(:,1)=1;


Xtrening=[XLtrening;XDtrening;XHtrening]; %Pakujemo sve u jedan trening
Ytrening=[YLtrening;YDtrening;YHtrening];
Xtest=[XLtest;XDtest;XHtest];             %Pakujemo sve u jedan test
Ytest=[YLtest;YDtest;YHtest];

ind1=randperm(length(Xtrening));   %Mesamo smestene podatke
ind2=randperm(length(Xtest));

Xtrening=Xtrening(ind1,:)';  %Dobijamo konacni izgled treninga
Ytrening=Ytrening(ind1,:)';

Xtest=Xtest(ind2,:)';        %Dobijamo konacni izgled testa
Ytest=Ytest(ind2,:)';

Ntrening=length(Xtrening);   %Broj odbiraka u trening i test skupu
Ntest=length(Xtest);
%% Krosvalidacija

%Inicijalizacija optimalnih hiperparametara
best_structure=[];
best_trainFcn = 'poslin';
best_reg= 0.2;
best_weight1 = 0;
best_weight2 = 0;
best_f1=0;
best_epoch=0;
% Pronalazenje optimalnih hiperparametara
for structure = {[7 10 7],[18 10 11], [5 5 4], [7 8 5]}
    for trainFcn = {'tansig','logsig','poslin'}
        for reg = {0.05, 0.1, 0.21}
            for weight1 = {2, 5, 8, 10}
                for weight2 = {2, 5, 8, 10}
                net = patternnet(structure{1});
                net.layers{1}.transferFcn = trainFcn{1};
                net.layers{2}.transferFcn = trainFcn{1};
                net.layers{3}.transferFcn = trainFcn{1};
                net.performParam.regularization = reg{1};
                net.divideFcn = 'divideind';
                net.divideParam.trainInd = 1:round(0.8*Ntrening);
                net.divideParam.valInd = round(0.8*Ntrening)+1:Ntrening;
                net.divideParam.testInd = [];               
                Xval = Xtrening(:, net.divideParam.valInd);
                Yval = Ytrening(:, net.divideParam.valInd);
                net.trainParam.max_fail = 6;
                net.trainParam.goal = 5e-6;
                net.trainParam.epochs = 350;
                net.trainParam.min_grad = 1e-8;
                net.trainparam.showWindow = false;
                w = ones(1,length(Xtrening));
                w(1,Ytrening(1,:)==1) = weight1{1};
                w(1,Ytrening(2,:)==1) = weight2{1};
                [net,tr] = train(net, Xtrening, Ytrening,[],[],w);
                Yval_pred = sim(net,Xval);
                [c,cm_val] = confusion(Yval,Yval_pred);
                cm_val = cm_val';
                recall1 = cm_val(1,1)/(cm_val(2,1)+cm_val(1,1)+cm_val(3,1));
                recall2 = cm_val(2,2)/(cm_val(2,2)+cm_val(1,2)+cm_val(3,2));
                recall3 = cm_val(3,3)/(cm_val(2,3)+cm_val(1,3)+cm_val(3,3));
                precision1 = cm_val(1,1)/(cm_val(1,2)+cm_val(1,1)+cm_val(1,3));
                precision2 = cm_val(2,2)/(cm_val(2,2)+cm_val(2,1)+cm_val(2,3));
                precision3 = cm_val(3,3)/(cm_val(3,2)+cm_val(3,1)+cm_val(3,3));
                f11=2*precision1*recall1/(precision1+recall1);
                f12=2*precision2*recall2/(precision2+recall2);
                f13=2*precision3*recall3/(precision3+recall3);
                f1=(f11+f12+f13)/3;  %Koristmo f1 score kao srednju vrednost sva tri f1 scora
                if (f1 > best_f1)
                    best_structure=structure{1};
                    best_trainFcn=trainFcn{1};
                    best_reg=reg{1};
                    best_epoch=tr.best_epoch;
                    best_weight1 = weight1{1};
                    best_weight2 = weight2{1};
                    best_f1=f1;
                end
             end              
          end
       end
   end
end


%% Realizacija neuralne mreze sa najboljim hiperparametrima

net = patternnet([best_structure]);            
net.layers{1}.transferFcn = best_trainFcn;
net.layers{2}.transferFcn = best_trainFcn;
net.layers{3}.transferFcn = best_trainFcn;
net.performParam.regularization = best_reg;
net.divideFcn = 'divideind';
net.divideParam.trainInd = [1:length(Xtrening)];
net.divideParam.valInd = [];
net.divideParam.testInd = [];

% Ocitavanje validacionih podataka
Xval = Xtrening(:, net.divideParam.valInd);
Yval = Ytrening(:, net.divideParam.valInd);
net.trainParam.max_fail = 6;
net.trainParam.goal = 1e-6;            
net.trainParam.epochs = best_epoch;
net.trainParam.min_grad = 1e-7;
net.trainparam.showWindow = true;            
w = ones(1,length(Xtrening));
w(1,Ytrening(1,:)==1)=best_weight1;
w(1,Ytrening(2,:)==1)=best_weight2;
[net,tr] = train(net, Xtrening, Ytrening,[],[],w);


%% Testiranje mreze

Ytrain_pred = sim(net,Xtrening);
Ytest_pred = sim(net,Xtest);

% Matrice konfuzije
figure()
plotconfusion(Ytrening, Ytrain_pred);
figure()
plotconfusion(Ytest, Ytest_pred);
%% Za prikaz klasa

NB=length(B);
figure(1); stem(1,NL); hold on;
stem(2,ND); hold on;
stem(3,NH); hold on;
stem(4,NB); hold off;
title('Klase u zavisnosti od broja odbiraka');
xlabel('Klasa');
ylabel('Broj odbiraka');
legend('Leptodactylidae','Dendrobatidae','Hylidae','Bufonidae');
xlim([-1 5]);
