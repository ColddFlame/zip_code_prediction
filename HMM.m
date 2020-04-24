function [Q_us_r,Q_ny_r,Q_us_o,Q_ny_o]=HMM

load data_1
f1=Fi;p1=points;
load data_2
f2=Fi;p2=points;
load data_3
f3=Fi;p3=points;
load data_4
f4=Fi;p4=points;
load data_5
f5=Fi;p5=points;
load W
load zipcode_data

%initial
for i = 1:size(zipcode,1)
    bit1(i) = floor(zipcode(i)/10000);
end
bit1ny = bit1(1:2205);
for i = 1:size(zipcode,1)
    bit2(i) = floor((zipcode(i)-bit1(i)*10000)/1000);
end
bit2ny = bit2(1:2205);
for i = 1:size(zipcode,1)
    bit3(i) = floor((zipcode(i)-bit1(i)*10000-bit2(i)*1000)/100);
end
bit3ny = bit3(1:2205);
for i = 1:size(zipcode,1)
    bit4(i) = floor((zipcode(i)-bit1(i)*10000-bit2(i)*1000-bit3(i)*100)/10);
end
bit4ny = bit4(1:2205);
for i = 1:size(zipcode,1)
    bit5(i) = zipcode(i)-bit1(i)*10000-bit2(i)*1000-bit3(i)*100-bit4(i)*10;
end
bit5ny = bit5(1:2205);
bit = [bit1',bit2',bit3',bit4',bit5'];
bitny = [bit1ny',bit2ny',bit3ny',bit4ny',bit5ny'];
%% Calculate parameters
% pi
for i = 0:9
    pi(i+1,:) = length(bit1(bit1(:)==i))/length(zipcode);
end
for i = 0:9
    piny(i+1,:) = length(bit1ny(bit1ny(:) == i))/2205;
end
% A
Atemp = zeros(10);
A = [];
for i=1:length(zipcode)
    for j=1:4
        Atemp(bit(i,j)+1,bit(i,j+1)+1)=Atemp(bit(i,j)+1,bit(i,j+1)+1)+1;
    end
end
for k = 1:10
    for j = 1:10 
    A(k,j)=Atemp(k,j)/sum(Atemp(k,:));
    end
end
Atempny = zeros(10);
Any = [];
for i = 1:2205
    for j = 1:4
        Atempny(bitny(i,j)+1,bitny(i,j+1)+1) = Atempny(bitny(i,j)+1,bitny(i,j+1)+1)+1;
    end
end
for k = 1:10
    for j = 1:10
    Any(k,j)=Atempny(k,j)/sum(Atempny(k,:));
    end
end
% B using rubine training data
f11 = FS(p1,size(p1,1));
f22 = FS(p2,size(p2,1));
f33 = FS(p3,size(p3,1));
f44 = FS(p4,size(p4,1));
f55 = FS(p5,size(p5,1));
wt = [56.099,151.46,469.08,325.44,-351.99,7.7442,21.582,2.1265,-1.6897,3.1999,-2.2073,0.078732,0.1621
    -27.921,-84.161,306.75,398.97,201.81,2.1093,-4.9326,-159.73,1.9195,1.8264,-1.484,0.28091,-6.4546
74.937,160.39,372.17,280.75,-85.085,34.391,-26.164,-24.97,3.3119,2.7538,-2.0667,0.34474,20.475
97.747,159.6,355.29,358.63,-133.81,14.148,-37.144,-26.451,5.2639,4.2483,-3.3225,0.24491,20.387
82.411,159.67,417.06,320.52,-265.75,33.273,24.784,6.989,1.5905,1.0245,-0.81414,0.22455,44.167
-42.286,8.1938,347.25,268.48,112.95,-8.7445,-12.479,-131.61,3.0011,1.6847,-1.1303,0.52087,55.127
-31.146,-65.946,296.73,318.86,21.564,1.5984,-5.9208,-72.155,-0.48515,3.5183,-2.2121,0.23392,-19.177
75.41,73.358,354.98,357.91,46.924,7.3686,-28.256,-106.33,2.7335,0.99566,-0.83618,0.42964,30.25
58.334,146.31,388.22,343.35,-331.86,29.706,-22.019,26.553,1.3939,4.5107,-3.1546,-0.016751,-12.548
39.136,155.63,397.67,328.85,6.5509,8.5129,-29.836,-87.316,-0.53726,2.8432,-2.1055,0.42904,27.122];
wt0 = [-468.8
    -407.59
-397.63
-457.84
-461.02
-340.23
-301.77
-379.33
-420.48
-439.82];
prob1=wt*f11+wt0;
prob2=wt*f22+wt0;
prob3=wt*f33+wt0;
prob4=wt*f44+wt0;
prob5=wt*f55+wt0;
for i = 1:size(prob1,1)
    for j = 1:size(prob1,2)
        if prob1(i,j) < 0
            prob1(i,j) = 0;
        end
    end
end
for i = 1:size(prob2,1)
    for j = 1:size(prob2,2)
        if prob2(i,j) < 0
            prob2(i,j) = 0;
        end
    end
end
for i = 1:size(prob3,1)
    for j = 1:size(prob3,2)
        if prob3(i,j) < 0
            prob3(i,j) = 0;
        end
    end
end
for i = 1:size(prob4,1)
    for j = 1:size(prob4,2)
        if prob4(i,j) < 0
            prob4(i,j) = 0;
        end
    end
end
for i = 1:size(prob5,1)
    for j = 1:size(prob5,2)
        if prob5(i,j) < 0
            prob5(i,j) = 0;
        end
    end
end
[~,num1]=max(prob1);
[~,num2]=max(prob2);
[~,num3]=max(prob3);
[~,num4]=max(prob4);
[~,num5]=max(prob5);
num = [num1-1,num2-1,num3-1,num4-1,num5-1];
Bt = [prob1/sum(prob1),prob2/sum(prob2),prob3/sum(prob3),prob4/sum(prob4),prob5/sum(prob5)];

% B
score1=w'*f1+w0';
score2=w'*f2+w0';
score3=w'*f3+w0';
score4=w'*f4+w0';
score5=w'*f5+w0';
[~,n1]=max(score1);
[~,n2]=max(score2);
[~,n3]=max(score3);
[~,n4]=max(score4);
[~,n5]=max(score5);
n = [n1-1,n2-1,n3-1,n4-1,n5-1];
B=[score1/sum(score1),score2/sum(score2),score3/sum(score3),score4/sum(score4),score5/sum(score5)];

%% problem 1 (us_rubine)
P = zeros(10,5);
for i=1:10
    P(i,1)=pi(i)*Bt(i,1);
end
[~,peak]=max(P(:,1));
Q(1,1) = peak-1;
for k = 2:5
for i = 1:10
    for j = 1: 10
        tempP(j,i) = P(i,k-1)*A(i,j)*Bt(j,k);
    end
end
P(:,k) = sum(tempP,2);
tempP = [];
[~,peak]= max(P(:,k));
Q(1,k) = peak-1;
end
Q_us_r = Q;%% outcome of us using project2 
%% problem 2(ny_rubine)
P = zeros(10,5);
for i=1:10
    P(i,1)=piny(i)*Bt(i,1);
end
[~,peak]=max(P(:,1));
Q(1,1) = peak-1;
for k = 2:5
for i = 1:10
    for j = 1: 10
        tempP(j,i) = P(i,k-1)*Any(i,j)*Bt(j,k);
    end
end
P(:,k) = sum(tempP,2);
tempP = [];
[~,peak]= max(P(:,k));
Q(1,k) = peak-1;
end
Q_ny_r = Q;%% outcome of ny using project2 
%% problem 3(us_orig)
P = zeros(10,5);
for i=1:10
    P(i,1)=pi(i)*B(i,1);
end
[~,peak]=max(P(:,1));
Q(1,1) = peak-1;
for k = 2:5
for i = 1:10
    for j = 1:10
        tempP(j,i) = P(i,k-1)*A(i,j)*B(j,k);
    end
end
P(:,k) = sum(tempP,2);
tempP = [];
[~,peak]= max(P(:,k));
Q(1,k) = peak-1;
end
Q_us_o = Q;%% outcome of us using w0 and w 
%% problem 4(ny_orig)
P = zeros(10,5);
for i=1:10
    P(i,1)=piny(i)*B(i,1);
end
[~,peak]=max(P(:,1));
Q(1,1) = peak-1;
for k = 2:5
for i = 1:10
    for j = 1:10
        tempP(j,i) = P(i,k-1)*Any(i,j)*B(j,k);
    end
end
P(:,k) = sum(tempP,2);
tempP = [];
[~,peak]= max(P(:,k));
Q(1,k) = peak-1;
end
Q_ny_o = Q;%% outcome of ny using w0 and w 
end
