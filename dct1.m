x1 = [10 11 12 11 12 13 12 11];
x2 = [10 -10 8 -7 8 -8 7 -7];
x12 = cat(2,x1,x2);

T8 = dctmtx(8);
T16 = dctmtx(16);

y1m = x1*T8';

y1 = dct(x1);
y2 = dct(x2);
y12 = cat(2,y1,y2);

y_16_12 = dct(x12);

figure
plot(y12)
title('8x8 DCT')
figure
plot(y_16_12)
title('16x16 DCT')