clc;
close all;
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %input image and pre processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                    
I = imread(input("enter the path of input image containing coins:"));
%displaying original input image and then converting it to gray
subplot 221
imshow(I);
title('original image');

I = rgb2gray(I);
[m,n] = size(I);
%displaying gray image
subplot 222
imshow(I);
title('grayscale image');

%adjusting intensity of the image according to code
%ASSUMPTION: background will always be white
for i = 1:m
    for j = 1:n
        I(i,j)=I(i,j)-220;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                       %GLOBAL THRESHOLDING:OTSU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%finding the histogram for GLOBAL THRESHOLDING
hist = imhist(I);
N=sum(hist); 
max_var=0; 

%Finding the normalized histogram i.e probability of each gray level
for i=1:256
    P(i)=hist(i)/N; 
end

%method: divide the gray levels in 2 classes based on the threshold
%class 0: from gray level 0 to threshold
%class 1: from gray level threshold+1 to 255
for T=2:255      
    %class 0 probability
    w0=sum(P(1:T)); 
    %class 1 probability
    w1=sum(P(T+1:256)); 
    %class 0 mean = u0
    u0=dot([0:T-1],P(1:T))/w0;
    %class 1 mean = u1
    u1=dot([T:255],P(T+1:256))/w1;
    %variance between the 2 classes: class 0 and class 1
    var=w0*w1*((u1-u0)^2); 
    %find the maximum variancce
    %the maximum variance classes will give the optimum global threshold
    %value. Hence classes 0 and 1 are selected such that they have max
    %variance and the T where they were divided is the threshold
    if var>max_var 
        max_var=var; 
        thr=T-1; 
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %BINARIZATION OF IMAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%using the threshold determined above by Osu's method, binarize the image
for i = 1:m
    for j = 1:n
        if (I(i,j) > thr)
            I(i,j) = 0;
        else
            I(i,j) = 255;
        end
    end
end
%displaying binary image
subplot 223
imshow(I);
title('binary image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                       %DILATION and EROSION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%new matrix called dil will be generated that is the dilated version of
%binary image generated above
dil = zeros(m,n);
%structuring element used below in dilation is '+'  is : %[1 1]
%the centre pix is the pivot element                     %[1 1]  

for i = 1 : m-1
    for j = 1 : n-1
        if I(i,j)==255 || I(i+1,j)==255 || I(i,j+1)==255 || I(i+1,j+1)==255
            dil(i,j)=255;
        end
    end    
end

%the dilated image may cause close coins to get connected hence, erode the
%dilated image to break unnecessary connections. ero is the new erosion of
%dilation image. The structuring element is: [1 1 1]
%the left top corner element is the pivot   %[1 1 1]
                                            %[1 1 1]
ero = dil;
for i = 1 : m-2
    for j = 1 : n-2
        ero(i,j) = min(min(dil(i:i+2,j:j+2)));
    end    
end

%displaying final erosion of dilated image
subplot 224
imshow(ero);
title('dilation and erosion');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %LABELLING REGOINS
               %i.e separating different coins
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%structuring element is B and A is the original image
B=[1 1 1; 1 1 1; 1 1 1];
A=imbinarize(ero);
%find the non-zero elements position from A and store in one_positions
one_positions = find(A==1);
%now taking the value of position in a temporary vaariable
one_positions = one_positions(1);
%label matrix that will demarkate different coins
L = zeros(size(A));
%counting the number of regoins
N=0;

%while all the positions of non-zero A elements are discovered
while(~isempty(one_positions))
    %each different component is labelled with a different label
    N=N+1;
    one_positions=one_positions(1);
    %X is another matrix that contains 1 at the same
    %position that is under consideration now from A matrix
    X = zeros(size(A));
    X(one_positions)=1;
    
    %updating the filter X for further expanding the regions
    Y = A & imdilate(X,B);
    
    %continue till al combinations are covered
    while(~isequal(X,Y)) 
        X=Y;
        Y=A&imdilate(X,B);
    end

    Pos=find(Y==1);
    A(Pos)=0;
    %Label the components
    L(Pos)=N;
    one_positions=find(A==1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %finding the no. of and the size(in pix) of each coin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n] = size(L);
%array to count number of pixs in each coin
coin = zeros(N,1);  
%finding the no. of pixs in each coin by counting the pixs in each regoin
for i = 1:m
    for j = 1:n
        if L(i,j)~=0
            num=L(i,j);
            coin(num)=coin(num)+1;
        end
    end
end
%printing the size of each coin in pixs
display('number of coins is:');
display(N);
display('size of each coin is given in pixels below:');
display(coin);

%displaying each coin separately and their size
for k = 1:N
    figure;
    L_single_coin = zeros(m,n);
    for i=1:m
        for j = 1:n
            if L(i,j) == k
                L_single_coin(i,j) = 255;
            end
        end
    end
    imshow(L_single_coin);
    title([num2str(coin(k)),'pixels']);
end        