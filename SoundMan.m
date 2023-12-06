
%Signal and Noise corruption
clear all; close all;
[y,fs] = audioread('SoundForce.m4a'); %HelloHowAreYou.wav
spch = audioplayer(y, fs); %sound(y,Fs);
play(spch)
N= length(y); %Data point
T= 1/fs; %T ระยะห่าง 
time= 0:T:(N-1)*T;
figure; plot(time, y); grid;
title('\bf Speech Waveform of "Signal & System"');
xlabel('\bf Time, seconds'); ylabel('\bf Amplitude, arbitary unit')

%code 2 %หาสเป็คตรัม
%Estimating DFT of speech signal 
N = length(y);
T = 1; % กำหนดค่า T ตามความเหมาะสมของคุณ

Y = T * fft(y); %Tเข้ามาช่วย
MY = abs(Y); %ค่าสมบูรณ์ แมกดิจูด(ขนาด) สเป็คตรัม
mag = MY(1:N/2);
fd = 1 / (N * T);
f = [0 : fd : (N/2 - 1) * fd];

figure;
plot(f, mag);
title('\bf MAGNITUDE SPECTRUM of "Signal & System" spoken word');
xlabel('\bf Frequency, Hz');
ylabel('\bf Magnitude, arb. unit');

%code3
[mxPk, Ind] = max(mag);
fo = f(Ind);
marker = zeros(size(mag)); % สร้างเวกเตอร์ขนาดเดียวกับ mag และเติมค่าศูนย์
marker(Ind) = mxPk;

figure;
plot(f, mag, 'b', f, marker, 'r'); % เพิ่มเวกเตอร์ marker เป็นสีแดงในกราฟ
title('\bf Detected FO Location');
xlabel('\bf Frequency, Hz');
ylabel('\bf Magnitude, arb. unit');
legend('Mag. Spec.', 'Marker'); % เพิ่มคำอธิบายสำหรับเส้นกราฟ

%code4
window_size = 10; % จำนวนจุดที่จะทำการเฉลี่ย
simple = movmean(mag, window_size); % คำนวณค่าเฉลี่ยเคลื่อนที่ใน Time Series

figure;
plot(f, mag, 'b', f, simple, 'r'); % แสดงกราฟ Mag. Spec. และ Smoothed Mag. Spec. พร้อมค่าต่างๆ
title('\bf Magnitude Spectrum and Smoothed Magnitude Spectrum');
xlabel('\bf Frequency, Hz');
ylabel('\bf Magnitude, arb. unit');
legend('Mag. Spec.', 'Smoothed');

% หาค่าสูงสุดใน Smoothed Mag. Spec.
[mxPkSm, IndSm] = max(simple);
foSm = f(IndSm);
fLg = fd * floor((window_size - 1) / 2); % ค่า Frequency Lag
fosm = foSm - fLg;

markerSm = zeros(size(simple)); % สร้างเวกเตอร์ marker ขนาดเท่ากับ simple
markerSm(IndSm) = mxPkSm;

figure;
plot(f, mag, 'b', f, marker, 'r', f, simple, 'g', f, markerSm, 'm');
legend('Mag. Spec.', 'Marker', 'Smoothed', 'MarkerSm');
title('\bf Detected FO Location and Smoothed Spectrum');
xlabel('\bf Frequency, Hz');
ylabel('\bf Magnitude, arb. unit');

%code5
%Showing both symmetric sides of Magnitude Spectrum
fdoub = [0:fd:(N-1)*fd]';
magdoub = MY(1:end);
figure; plot(fdoub, magdoub);
title('\bf MAGNITUDE SPECTRUM of "Signal & System" spoken word'); 
xlabel('\bf Frequency, Hz'); ylabel('\bf Magnitude, arb. unit')

%code6
%Shifting spectral components to center respective to frequency axis
magdoubShift= fftshift(magdoub); fsR= [0:fd:(N/2-1)*fd];
fsL= [(-(N/2-1):1:0)*fd]; fdoubShift= [fsL fsR];
figure, plot(fdoubShift,magdoubShift)
title('\bf Shifted Magnitude Spectrum');
xlabel('\bf Frequency, Hz'); ylabel('\bf Magnitude, arb. unit')

%code7
fs1= fs;
fs2= 10000; %10KHz
y2= resample(y, fs2, fs1);
N2= length(y2);
T2= 1/fs2;
time2= [0:T2:(N2-1)*T2];
figure; plot(time, y,'r',time2, y2,'b:'); grid; 
legend('Original ','Resampled')

%code8 มีปัญหา
% Estimating DFT of Resampled speech signal
Y2 = T2 * fft(y2);
MY2 = abs(Y2);
mag2 = MY2(1:N2/2); % Only the first half b/c the remainder is redundant 
fd2 = 1 / (N2 * T2);
f2 = [0 : fd2 : (N2/2 - 1) * fd2];
figure;
plot(f, mag, 'r', f2, mag2, 'b'); grid on;
title('\bf Comparison of MAGNITUDE SPECTRUMs');
xlabel('\bf Frequency, Hz');
ylabel('\bf Magnitude, arb. unit');
legend('@fs1', '@fs2');

%code9
%STFT :Short-time Fourier Tranferform - Time vs Frequency Spectrum 
figure,
%specgram(data, Nfft, fs2, window(Nfft), %overlapping)
specgram(y2, 2^8, fs2,blackman(2^8), round(0.5*2^8));
figure,
%spectrogram(data, window(Nfft), %overlapping), Nfft, fs2);
spectrogram(y2,blackman(2^8), round(0.5*2^8), 2^8, fs2);
%Try own record of 5-10Seconds

%code10
%Speech corrupted with noise signal
ns = randn(size(time2)); % random signal
wn= 0.15*ns'
y2n = y2 + wn;
spch2 = audioplayer(y2n, fs2); %sound(y,Fs);
play(spch2)
figure;
%subplot(112);
plot(time2, y2n, 'r', time2, wn, 'g', time2, y2, 'b'); grid;
title('\bf Corrupted Speech Waveform with Random Noise'); 
xlabel('\bf Time, seconds'); 
ylabel('\bf Amplitude, arbitary unit') 
legend('Corrupted speech','Noise','Speech');

%code11
[smax ix] = max(y2);
[smin, in] = min(y2);
[wmax ix] = max(wn);
[wmin in] = min(wn);
[snmax ix] = max(y2n);
[snmin in] = min(y2n);
x= (snmin:.05:snmax);
[n1,b1]= hist(wn,length(x));
[n2,b2]= hist(y2,length(x));
[n3,b3]= hist(y2n,length(x));
figure, bar(b1,n1,'r');
hold on;
bar(b2,n2,'b');
hold on;
bar(b3,n3,'g');
hold off;
legend('Noise', 'Speech','Corrupted speech');

%code12
%Unimodal pdf of signals
bin=30; 
norm1 = csnormp(b1, mean(wn),var (wn)); 
norm2 = csnormp(b2, mean(y2),var (y2)); 
norm3 = csnormp(b3, mean(y2n),var(y2n)); 
figure; plot(b1, norm1, 'r', 'linewidth',3); hold on; plot(b2, norm2, 'b', 'linewidth',3); 
hold on; plot(b3, norm3, 'm', 'linewidth',3) 
set(gca, fontsize', 14, 'fontweight', 'bold'); set(gca, 'fontname', 'helvetica', 'linewidth',3); 
legend('\bf wn', '\bf signal', '\bf y2n'); 
xlabel('Probability', 'fontweight', 'bold', 'fontsize',14);ylabel('\bf Number', 'fontsize',14); 
set(gca, 'color', [1 1 1]); % sets background color of plot hold off;
hold off;

