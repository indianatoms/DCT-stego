clc; clear all;
%X = my_dct_mtx(8);
T8 = dctmtx(8);
%randomly_generated_vector = rand(1,8)*10;
I = double(imread(['lenna.tif']));
%figure;
%imshow(uint8(I))

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



scale = 0.1;
fun = @(block_struct) my2DDCT(block_struct.data);
X1 = blockproc(I,[8 8],fun);
fun1 = @(block_struct) quantizeMatrix(block_struct.data,scale);
QX2 = blockproc(X1,[8 8],fun1);

%hidden message
s = 'secet';
if mod(strlength(s),2) == 1
    s = strcat(s,',')
end

%get binaray representation of hidden message
binary = reshape(dec2bin(s)',1,[]);

%encode the message usign LSB of DC bands
%COMMENT OUT FOR LSB
%stegoQ = stego(QX2,binary);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DCT M-3 IMPLEMENTATION
k = [1,2];
stegoQ = stegoM3(QX2,binary,k);
%%%%%%%%%%%%%%%%%%%%%%%5

%make zigzag of each image
zigzag = @(block_struct) ZigZagscan(block_struct.data);
ZZmat = blockproc(stegoQ,[8 8],zigzag);
ZZvec = reshape(ZZmat',[1,262144]);
%huffman encoding of Zigzag
symbols = unique(ZZvec(:));
counts = hist(ZZvec(:), symbols);
p = double(counts) ./ sum(counts);
[dict,avglen] = huffmandict(symbols,p);
hcode = huffmanenco(ZZvec(:),dict);
%dcode huff
hdecode = huffmandeco(hcode,dict);
%unzigzag 
decStegoQ = unzigzag(hdecode);
decBlockStegoQ = reshape(permute(reshape(decStegoQ,size(decStegoQ,1),512,[]),[1,3,2]),[],512);

%%% For M3
x = unM3msg(decBlockStegoQ,k);
disp(x);
%%% For LSB 
%x = LSBS(decBlockStegoQ);


hidden = message(x);
disp(hidden);



%fun3 = @(block_struct) unscaledQuantizeMatrix(block_struct.data,scale);
%UnscaledQX2 = blockproc(decBlockStegoQ,[8 8],fun3);
%fun2 = @(block_struct) myInverse2DDCT(block_struct.data);
%I2 = blockproc(decBlockStegoQ,[8 8],fun2);
%figure;
%imshow(uint8(I2))


function y = unzigzag(hdecode)
    %rate = zeros(8,8);
    for i = 0:64:length(hdecode)-64 
        temp = invzigzag(hdecode(i+1:i+64),8,8);
        %disp(size(temp));
        if (i==0)
            rate = temp;
        else
            rate = horzcat(rate,temp);
        end
    end
    y = rate;
end

function y = message(X)
str = "";
    for r = 0:7:length(X)-7
       str = append(str,char(bin2dec(num2str(X(r+1:r+7)))));
    end
    y = str; 
end

function y = LSBS(N)
    x = []; 
    for r = 1:8:size(N,1)
        LSB = mod(double(N(r,r)), 2);
        %disp(LSB);
        x(end+1) = LSB;
    end
    y = x;
end

function y = stego(N,pattern)
    i = 1;
    for r = 1:length(pattern)
        %disp(pattern(r));
        %disp(dec2bin(N(i,i)));
        N(i,i) = bitset(N(i,i),1,str2double(pattern(r)));
        %disp(dec2bin(N(i,i)));
        i = 1 + r * 8;
    end
    y = N;
end

function y = PSNR(q,ref)
    y = 10*log10((255^2/immse(q,ref)));
end


%make this after DCt
function quanMat = quantizeMatrix(N,scale)
    q_mtx = scale * [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62; 18 22 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    quanMat = fix(fix(N ./ q_mtx).*q_mtx);
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

function out=invzigzag(in,num_rows,num_cols)
    % Inverse Zig-zag scanning
    % This function reorders a 1-D array into a specifiable
    % 2-D matrix by implementing the INVERSE ZIG-ZAG SCNANNING procedure.
    % IN specifies the input 1-D array or vector
    % NUM_ROWS is the number of rows desired in the output matrix
    % NUM_COLS is the number of columns desired in the output matrix
    % OUT is the resulting inverse zig-zag scanned matrix
    % having the same number of elements as vector IN
    %
    % As an example,
    % IN = [1     2     4     7     5     3     6     8    10    11     9    12];
    % OUT = INVZIGZAG(IN,4,3)
    % OUT=
    %	1     2     3
    %	4     5     6
    %	7     8     9
    %	10    11    12
    %
    %
    % Oluwadamilola (Damie) Martins Ogunbiyi
    % University of Maryland, College Park
    % Department of Electrical and Computer Engineering
    % Communications and Signal Processing
    % 22-March-2010
    % Copyright 2009-2010 Black Ace of Diamonds.
    tot_elem=length(in);
    if nargin>3
        error('Too many input arguments');
    elseif nargin<3
        error('Too few input arguments');
    end
    % Check if matrix dimensions correspond
    if tot_elem~=num_rows*num_cols
        error('Matrix dimensions do not coincide');
    end
    % Initialise the output matrix
    out=zeros(num_rows,num_cols);
    cur_row=1;	cur_col=1;	cur_index=1;
    % First element
    %out(1,1)=in(1);
    while cur_index<=tot_elem
        if cur_row==1 & mod(cur_row+cur_col,2)==0 & cur_col~=num_cols
            out(cur_row,cur_col)=in(cur_index);
            cur_col=cur_col+1;							%move right at the top
            cur_index=cur_index+1;

        elseif cur_row==num_rows & mod(cur_row+cur_col,2)~=0 & cur_col~=num_cols
            out(cur_row,cur_col)=in(cur_index);
            cur_col=cur_col+1;							%move right at the bottom
            cur_index=cur_index+1;

        elseif cur_col==1 & mod(cur_row+cur_col,2)~=0 & cur_row~=num_rows
            out(cur_row,cur_col)=in(cur_index);
            cur_row=cur_row+1;							%move down at the left
            cur_index=cur_index+1;

        elseif cur_col==num_cols & mod(cur_row+cur_col,2)==0 & cur_row~=num_rows
            out(cur_row,cur_col)=in(cur_index);
            cur_row=cur_row+1;							%move down at the right
            cur_index=cur_index+1;

        elseif cur_col~=1 & cur_row~=num_rows & mod(cur_row+cur_col,2)~=0
            out(cur_row,cur_col)=in(cur_index);
            cur_row=cur_row+1;		cur_col=cur_col-1;	%move diagonally left down
            cur_index=cur_index+1;

        elseif cur_row~=1 & cur_col~=num_cols & mod(cur_row+cur_col,2)==0
            out(cur_row,cur_col)=in(cur_index);
            cur_row=cur_row-1;		cur_col=cur_col+1;	%move diagonally right up
            cur_index=cur_index+1;

        elseif cur_index==tot_elem						%input the bottom right element
            out(end)=in(end);							%end of the operation
            break										%terminate the operation
        end
    end

end

function y = stegoM3(N, msg, k)
i = 1;
for r = 1:2:length(msg)
    pair = strcat(msg(r), msg(r + 1));
    disp(pair);
    first = N(i, i + k(1));
    second = N(i, i + k(2));
    disp(i+k(1))
    disp(i+k(2))
    disp(first)
    disp(second)
 
    if mod((first - second), 3) == 0
        disp("mod0")
        if mod((first), 2) == 0
            if pair == "01"
                disp("here")
                N(i, i + k(1)) = first + 1;
            elseif pair == "10"
                N(i, i + k(2)) = second + 1;
            elseif pair == "11"
                N(i, i + k(1)) = first + 1;
                N(i, i + k(2)) = second + 1;
            end
        else
            if pair == "00"
                N(i, i + k(1)) = first + 1;
                N(i, i + k(2)) = second + 1;
            elseif pair == "01"
                N(i, i + k(1)) = first + 1;
            elseif pair == "10"
                N(i, i + k(2)) = second + 1;
            end
        end
    elseif mod((first - second), 3) == 1
        %disp("mod1")
        if mod((first), 2) == 0
            if pair == "00"
                N(i, i + k(2)) = second + 1;
            elseif pair == "10"
                N(i, i + k(1)) = first + 1;
            elseif pair == "11"
                N(i, i + k(1)) = first - 1;
            end
        else
            if pair == "00"
                N(i, i + k(1)) = first - 1;
            elseif pair == "10"
                N(i, i + k(1)) = first + 1;
            elseif pair == "11"
                N(i, i + k(2)) = second + 1;
            end
        end
    elseif mod((first - second), 3) == 2
        %disp("mod2")
        if mod((first), 2) == 0
            if pair == "00"
                N(i, i + k(2)) = second - 1;
            elseif pair == "01"
                N(i, i + k(1)) = first - 1;
            elseif pair == "11"
                N(i, i + k(1)) = first + 1;
            end
        else
            if pair == "00"
                N(i, i + k(1)) = first + 1;
            elseif pair == "01"
                N(i, i + k(1)) = first - 1;
            elseif pair == "11"
                N(i, i + k(2)) = second - 1;
            end
        end
    end 
    i = i + 8;
end
y = N;
end

function y = unM3msg(N,k)
    x = []; 
    for r = 1:8:size(N,1)
        first = N(r,r + k(1));
        second = N(r,r + k(2));
        if mod((first - second), 3) == 0
            if mod((first), 2) == 0
                x(end+1) = 0;
                x(end+1) = 0;
            else
                x(end+1) = 1;
                x(end+1) = 1;
            end
        elseif mod((first - second), 3) == 1
            x(end+1) = 0;
            x(end+1) = 1;
        elseif mod((first - second), 3) == 2  
            x(end+1) = 1;
            x(end+1) = 0;
        end
        
    end
    y = x;
end

