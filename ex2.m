clc;
%X = my_dct_mtx(8);
T8 = dctmtx(8);
%randomly_generated_vector = rand(1,8)*10;
I = double(imread(['lenna.tif']));
figure;
imshow(uint8(I))

%task1
% y1m = randomly_generated_vector*X'
% y1 = dct(randomly_generated_vector);

%task2
%y1m = my2DDCT(randomly_generated_matrix)
%y1 = dct2(randomly_generated_matrix)

%task4
%rgm = myInverse2DDCT(y1)

%task5
% one8x8 = uint8(255 * rand(8));
% imshow(one8x8, []);
% one8x8 = one8x8 - 128;
% X1 = dct2(one8x8);
% T = dctmtx(8);

scales = 0.01:0.05:1;
rate = zeros(1,length(scales));
quailty = zeros(1,length(scales));
for r = 1:length(scales)
    scale = scales(r);
    fun = @(block_struct) my2DDCT(block_struct.data);
    X1 = blockproc(I,[8 8],fun);
    fun1 = @(block_struct) quantizeMatrix(block_struct.data,scale);
    QX2 = blockproc(X1,[8 8],fun1);
    fun3 = @(block_struct) unscaledQuantizeMatrix(block_struct.data,scale);
    UnscaledQX2 = blockproc(X1,[8 8],fun3);
    fun2 = @(block_struct) myInverse2DDCT(block_struct.data);
    I2 = blockproc(QX2,[8 8],fun2);
    rate(r) = entropy(UnscaledQX2);
    quailty(r) = PSNR(I,I2);
%     figure;
%     imshow(uint8(I2));
end

figure;
plot(rate,quailty)



%imshow(randomly_generated_matrix);
%only the other frequencies are chagned but not the DC band. As the 

function y = PSNR(q,ref)
    y = 10*log10((255^2/immse(q,ref)));
end


%make this after DCt
function quanMat = quantizeMatrix(N,scale)
    q_mtx = scale * [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62; 18 22 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    quanMat = fix(N ./ q_mtx).*q_mtx;
end

function unscaleQuanMat = unscaledQuantizeMatrix(N,scale)
    q_mtx = scale * [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62; 18 22 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    unscaleQuanMat = fix(N ./ q_mtx);
end

function matrix  = my_dct_mtx(N)
    matrix = zeros(N);
    for r = 1:size(matrix,1)    % for number of rows of the image
        for c = 1:size(matrix,2)    % for number of columns of the image
            if r==1
                v = sqrt(1/N)*cos(((2*(c-1)+1)*(r-1)*pi)/(2*N));
            else
                v = sqrt(2/N)*cos(((2*(c-1)+1)*(r-1)*pi)/(2*N));
            end
            matrix(r,c) =  v;
        end
    end
end

function y = my2DDCT(img)
%task3
%SIMPLIFIED VERSION
    X = my_dct_mtx(8);
    y = X*img*X';
%LONG VERSION
%     for r = 1:size(img,1)    % for number of rows of the image
%         img(r,:) = img(r,:)*X';
%     end
%     for c = 1:size(img,2)   % for number of columns of the image
%         img(:,c) = img(:,c)'*X';
%     end
%     y = img;
end

function img = myInverse2DDCT(y)
%SIMPLIFIED VERSION
    X = my_dct_mtx(8);
    img = X\y*inv(X');
end
     



