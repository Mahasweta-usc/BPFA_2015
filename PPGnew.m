% Grayscale image denoising running file for the paper:
% "Nonparametric Bayesian Dictionary Learning for Analysis of Compressive
% or Incomplete Image Measurements," submitted.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu,
% mingyuan.zhou@duke.edu
% Verison 0: 06/14/2009
% Version 1: 09/20/2009
% Version 2: 10/21/2009
% Version 3: 10/28/2009
% Version 4: 11/03/2009
% Last update: 03/30/2010

clear all
close all
%% Define
filename = 'C:\Users\deel\Desktop\competition_data\competition\DATA_02_TYPE02.mat';
filename_ground =  'C:\Users\deel\Desktop\competition_data\competition\DATA_02_TYPE02_BPMtrace.mat';
load(filename);
load(filename_ground);
Fs = 125;
[b,a] = ellip(6, 0.5, 40 ,[0.8/(Fs/2), 5/(Fs/2)]);
fvtool(b,a);
%freqz(b,a, 256);
%% Input
input_signal = sig(2,:);
figure(2)
subplot(3,1,1)
plot(input_signal(1:1000));
subplot(3,1,2)
plot(-Fs/2:Fs/4096:(Fs/2)-(Fs/4096),fftshift(abs(fft(input_signal(1:1000),4096))));
subplot(3,1,3)
[pxx,f] = periodogram(input_signal(1:1000),[],4096, 125);
plot(f,pxx);
%% Output
output_signal = filter(b,a,input_signal);
figure(3)
subplot(3,1,1)
plot(output_signal(1:1000));
subplot(3,1,2)
plot(-Fs/2:Fs/4096:(Fs/2)-(Fs/4096),fftshift(abs(fft(output_signal(1:1000),4096))));
subplot(3,1,3)
[pxx,f] = periodogram(output_signal(1:1000),[],4096, 125);
plot(f,pxx);
IMin = input_signal;
IMin0 = input_signal;
% %% Added Part
% numfiles=8;
% path = 'C:\Users\Sandeep Dsouza\Downloads\Acads\energy desegg\low_freq\house_2\channel_';
% No_of_Seconds = 500;
% X=train_kolter(path, No_of_Seconds, numfiles);
% D_full = [];
% D_size = [];
% %% Original
% for k = 1:numfiles
PatchSize = 8; %patch size
sigma = 0; %25; %noise stand deviation
% IMin0 = X{1,k};
% IMin  = X{1,k};
IMname = 'house'; %pepper256,barbara,lena512,boat,fingerprint,
% 
% IMin0 = imread([IMname,'.png']);
% %IMin0 = imresize(IMin0,0.5);
% 
% IMin0=im2double(IMin0);
% randn('seed',0)
% IMin = IMin0 + sigma/255*randn(size(IMin0));
%PSNRIn = -10*log10((mean((IMin(:)-IMin0(:)).^2)));
IterPerRound = ones(PatchSize,PatchSize); %Maximum iteration in each round
IterPerRound(end,end) = 200;

if size(IMin,1)*size(IMin,2)<=300*300
    K = 256;  %dictionary size
else
    K = 512;
end

K = 50;

DispPSNR = true; %Calculate and display the PSNR or not;

ReduceDictSize = false; %Reduce the ditionary size during training if it is TRUE, can be used to reduce computational complexity 
IsSeparateAlpha = false; %use a separate precision for each factor score vector if it is TRUE. 
InitOption = 'Rand'; %Initialization with 'SVD' or 'Rand'
LearningMode = 'online'; %'online' or 'batch'

[D,S,Z,BPM] = BPFA_Denoise_PPG_sep(IMin, PatchSize, K, DispPSNR, IsSeparateAlpha, InitOption, LearningMode, IterPerRound, ReduceDictSize, IMin0, sigma, IMname, BPM0, sig);
        
% b = full(Z);
% c = sum(b);
% figure(k)
% stem(c)
%PSNROut = PSNRave;
%D_size = [D_size, size(D,2)];
% if(K - size(D,2) ~= 0)
%     temp_dict = zeros(size(D,1),K - size(D,2));
%     D = [D,temp_dict];
% end
%D_full = [D_full,D];
% figure;
% subplot(1,3,1); imshow(IMin0); title('Original clean image');
% subplot(1,3,2); imshow(IMin); title((['Noisy image, ',num2str(PSNRIn),'dB']));
% subplot(1,3,3); imshow(ave.Iout); title((['Denoised Image, ',num2str(PSNROut),'dB']));
% print(gcf,'-dpng',[IMname '_' num2str(sigma) '_results.png'])
% imwrite(ave.Iout, [IMname '_' num2str(sigma) '.png']);
% figure;
% [temp, Pidex] = sort(sum(Z,1),'descend');
% Dsort = D(:,Pidex);
% I = DispDictionary(Dsort);
% imwrite(I, [IMname '_' num2str(sigma) '_dict.png']);
% title('The dictionary trained on the noisy image');

