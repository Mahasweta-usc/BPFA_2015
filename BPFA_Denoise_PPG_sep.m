function [D,S,Z,BPM] = BPFA_Denoise_PPG_sep(IMin, PatchSize, K, DispPSNR, IsSeparateAlpha, InitOption, LearningMode, IterPerRound,ReduceDictSize, IMin0, sigma, IMname, BPM0, sig,BPM)
D = [];
S = [];
Z = [];
Iout = [];
PSNR = 0;
PSNEave = 0;

%------------------------------------------------------------------
% The BPFA grayscale image denoising program for the paper:
% "Non-Parametric Bayesian Dictionary Learning for Sparse Image
% Representations," Neural Information Processing Systems (NIPS), 2009.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu
% Version 0: 06/14/2009
% Version 1: 09/13/2009
% Version 2: 10/21/2009
% Version 3: 10/28/2009
% Version 4: 11/03/2009
% Updated in 03/30/2010
%
%------------------------------------------------------------------
% Input:
%   IMin: noisy image.
%   PatchSize: patch size, 8 is commonly used.
%   K: predefined dictionary size.
%   DispPSNR: calculate and display the instant PSNR during learning if it
%   is TRUE
%   IsSeparateAlpha: use a separate precision for each factor score vector
%   if it is TRUE.
%   InitOption? 'SVD' or 'Rand';
%   LearningMode: 'online' or 'batch';
%   IterPerRound: maximum iteration in each round.
%   ReduceDictSize: reduce the dictionary size during learning if it is
%   TRUE
%   IMin0: original noise-free image (only used for PSNR calculation and
%   would not affect the denoising results).
%   sigma: noise variance (has no effect on the denoising results).
%   IMname: image name (has no effect on the deoising results).

% Output:
%   ave: denoised image (averaged output)
%   Iout: denoised image (sample output)
%   D: dictionary learned from the noisy image
%   S: basis coefficients
%   Z: binary indicators for basis usage
%   idex: patch index
%   Pi: the probabilities for each dictionary entries to be used
%   NoiseVar: estimated noise variance
%   alpha: precision for S
%   phi: noise precision
%   PSNR: peak signal-to-noise ratio
%   PSNRave: PSNR of ave.Iout
%------------------------------------------------------------------

if nargin < 2
    PatchSize=8;
end
if nargin < 3
    K=256;
end
if nargin < 4
    DispPSNR = true;
end
if nargin < 5
    IsSeparateAlpha = false;
end
if nargin < 6
    InitOption = 'SVD';
end
if nargin < 7
    LearningMode = 'online';
end
if nargin < 8
    IterPerRound = ones(PatchSize,PatchSize);
    IterPerRound(end,end) = 50;
end
if nargin < 9
    ReduceDictSize = false;
end
if nargin < 10
    IMin0=IMin;
end
if nargin < 11
    sigma=[];
end
if nargin < 12
    IMname=[];
end

sizeIMin = size(IMin);
idex=[];
PSNR=[];
NoiseVar=[];
X_k=[];
PSNRave = 0;

% Set Hyperparameters
c0=1e-4;
d0=1e-4;
e0=1e-4;
f0=1e-4;

ave.Iout = zeros(size(IMin));
ave.Count = 0;
Fs = 125;
No_of_Seconds = 8;
No_of_Training = floor(size(IMin,2)/125)-(No_of_Seconds-1)
predicted_hb = zeros(No_of_Training,1);
original_hb = zeros(No_of_Training,1);
X_k = [];
DispPSNR = false;
[b,a] = ellip(6, 0.5, 40 ,[0.8/(Fs/2), 5/(Fs/2)]);
if strcmp(LearningMode,'online')==1
%     for colj=1:PatchSize
%  for rowi=1:PatchSize
%             idexold = idex;
%             idexNew = idexUpdate(sizeIMin,PatchSize,colj,rowi);
%             idex = [idexold;idexNew];
%             X_k = Update_Input(X_k,IMin,idexNew,PatchSize);
%             [P,N] = size(X_k)
%             pause
            %Sparsity Priors
      step_size = 2;
      count = 1;
      temp_X_k = [];
      BPM = [];
      for no = 1:step_size:No_of_Training
            no
            %pause
            if strcmp(InitOption,'SVD')==1
                a0=1;
                %b0=N/8;
                b0 = 10;
            else
                a0=1;
                b0=100;
            end
            heart_range = [(BPM0(count,1)-10)/60,BPM0(count,1)/60,(BPM0(count,1)+10)/60];
            if count == 2
                trend = 0;
            end
            if count ~=2 && no ~=1
               % trend = BPM(count-1,1)-BPM(count-2,1);
            end
            if no == 1
                freq_range = round(heart_range*4096/(Fs));
                prev_heart_range = [(BPM0(count,1)-10),BPM0(count,1),(BPM0(count,1)+10)];
                prev_harmonic_range =  [2*(BPM0(count,1)-10),2*BPM0(count,1),2*(BPM0(count,1)+10)];
                prev_range = freq_range;
                prev_harm_range = round(prev_harmonic_range*4096/(60*Fs));
            else
                prev_heart_range = [(BPM(count-1,1)-10),BPM(count-1,1),(BPM(count-1,1)+10)];
                prev_harmonic_range =  [2*(BPM(count-1,1)-10),2*BPM(count-1,1),2*(BPM(count-1,1)+10)];
                prev_range = round(prev_heart_range*4096/(60*Fs));
                prev_harm_range = round(prev_harmonic_range*4096/(60*Fs));
            end
            %heart_range = [(BPM0(count,1)-10)/60,BPM0(count,1)/60,(BPM0(count,1)+10)/60];
            X_k_in = IMin((no-1)*125 + 1: no*125 + (No_of_Seconds-1)*125);
            acc_x = sig(4, (no-1)*125 + 1: no*125 + (No_of_Seconds-1)*125)';
            acc_y = sig(5, (no-1)*125 + 1: no*125 + (No_of_Seconds-1)*125)';
            acc_z = sig(6, (no-1)*125 + 1: no*125 + (No_of_Seconds-1)*125)';
            X_k_trans = filter(b,a,IMin((no-1)*125 + 1: no*125 + (No_of_Seconds-1)*125));
            X_k = X_k_trans';
            %correct_frequency = round(BPM0(count,1)*4096/(60*Fs));
            upper = ceil(5/((Fs/4096)));
            figure(4)
            subplot(3,1,1)
            plot(X_k(1:1000));
            subplot(3,1,2)
            plot(-Fs/2:Fs/4096:(Fs/2)-(Fs/4096),fftshift(abs(fft(X_k(1:1000),4096))));
            subplot(3,1,3)
            [pxx,f] = periodogram(X_k(1:1000),[],4096, 125);
            plot(f(1:upper+1),pxx(1:upper+1)/max(pxx(1:upper+1)),'b',heart_range,zeros(1,3),'b*','LineWidth',2);
            hold on
%             X_k = [];
%             X_k = pxx(1:upper+1);
            [pxx,f] = periodogram(acc_x(1:1000),[],4096, 125);
            plot(f(1:upper+1),pxx(1:upper+1)/max(pxx(1:upper+1)),'r','LineWidth',2);
            [pxx,f] = periodogram(acc_y(1:1000),[],4096, 125);
            plot(f(1:upper+1),pxx(1:upper+1)/max(pxx(1:upper+1)),'g','LineWidth',2);
            [pxx,f] = periodogram(acc_z(1:1000),[],4096, 125);
            plot(f(1:upper+1),pxx(1:upper+1)/max(pxx(1:upper+1)),'black','LineWidth',2);
            hold off
            diff_step = 3;
%            X_k = (filter(b,a,IMin((no-1)*125 + 1+diff_step : no*125 + (No_of_Seconds-1)*125 + diff_step)))'-X_k;
            %X_k = [X_k, IMin((no-1)*125 + 1: no*125 + (No_of_Seconds-1)*125)'];
%             size(X_k)
            %Initializations for new added patches
           if no == 1 %if rowi==1 && colj==1
                %Random initialization
                [D,S,Z,phi,alpha,Pi] = InitMatrix_Denoise(X_k,K,InitOption,IsSeparateAlpha,IMin);  
          % else
%                 %Initialize new added patches with their neighbours
            %if no ~= 1
                %S = [S;S(end,:)];
                %Z = [Z;Z(end,:)];
           else
                if(no<=20)
                   S = [S;S(end,:)];
                   Z = [Z;Z(end,:)];
                else
                   S = [S(2:end,:);S(end,:)];
                   Z = [Z(2:end,:);Z(end,:)];
                end
                 
            end
%                 %[S,Z] = SZUpdate(S,Z,rowi,idexNew,idexold);
%             end
%             idext = N-size(idexNew,1)+1 : N;
%             %X_k(:,idext) = X_k(:,idext) - D*S(idext,:)';
%             X_k(:,idext) = X_k(:,idext) - D*(S(idext,:).*Z(idext,:))';
            %X_k(:,end) = X_k(:,end) - D*(S(end,:).*Z(end,:))';
            
            signal = X_k;
            X_k = X_k - D*(S(end,:).*Z(end,:))';
            if(no<=20)
                temp_X_k = [temp_X_k, X_k];
            else
                temp_X_k = [temp_X_k(:,2:end),X_k];
            end
            maxIt = 1000;%IterPerRound(colj,rowi);
%             if no == No_of_Training
%                 maxIt = 200;
%             end
            
            for iter=1:maxIt
                tic
                %Sample D, Z, and S
                if size(IMin,3)==1
                    Pi(1) = 1;
                end
                [temp_X_k, D, Z, S] = SampleDZS(temp_X_k, D, Z, S, Pi, alpha, phi, true, true, true);
                %Sample Pi
                Pi = SamplePi(Z,a0,b0);
                %Sample alpha
                alpha = Samplealpha(S,e0,f0,Z,alpha);
                %Sample phi
                phi = Samplephi(temp_X_k,c0,d0);
                ittime=toc;
                
                NoiseVar(end+1) = sqrt(1/phi)*255;
                if ReduceDictSize && colj>2
                    sumZ = sum(Z,1)';
                    if min(sumZ)==0
                        Pidex = sumZ==0;
                        D(:,Pidex)=[];
                        K = size(D,2);
                        S(:,Pidex)=[];
                        Z(:,Pidex)=[];
                        Pi(:,Pidex)=[];
                        alpha(Pidex)=[];
                    end
                end
                
%                 if DispPSNR==1 || (rowi==PatchSize&&colj==PatchSize)
%                     if rowi==PatchSize
%                         %Iout    =   DenoiseOutput(D*(S.*Z)',sizeIMin,PatchSize,idex,MuX);
%                         Iout    =   DenoiseOutput_LowMemoryReq(D,S,sizeIMin,PatchSize,idex);
%                         if colj==PatchSize
%                             ave.Count = ave.Count + 1;
%                             if ave.Count==1
%                                 ave.Iout = Iout;
%                             else
%                                 ave.Iout= 0.85*ave.Iout+0.15*Iout;
%                             end
%                             PSNRave = -10*log10((mean((ave.Iout(:)-IMin0(:)).^2)));
%                         end
%                         PSNR(end+1) = -10*log10((mean((Iout(:)-IMin0(:)).^2)));
%                         if rowi==PatchSize && mod(iter,10)==1
%                             disp(['round:',num2str([colj,rowi]),'    iter:', num2str(iter), '    time: ', num2str(ittime), '    ave_Z: ', num2str(full(mean(sum(Z,2)))),'    M:', num2str(nnz(mean(Z,1)>1/1000)),'    PSNR:',num2str(PSNR(end)),'    PSNRave:',num2str(PSNRave),'   NoiseVar:',num2str(NoiseVar(end)) ])
%                             %save([IMname,'_Denoising_',num2str(sigma),'_',num2str(colj),'_',num2str(rowi)], 'Iout','D','PSNR','Pi','IMin','IMin0','phi','alpha','NoiseVar','idex','ave');
%                         end
%                     end
%                 end
            end
            
            figure(6)
            numel = nnz(Z(end,:));
            [row,col] = find(Z(end,:) == 1);
            maxima = [];
            heartbeat = [];
            heartbeat_temp = [];
            fft_size = 4096;
            temp_S = full(S);
            filter_size = 40;
            filter_ma = (1/20)*ones(1,20);
            
            sort_pks=[];
            all_har_pks=[];
            heart_list=[];
            temp_corr = [];
            for k = 1:numel
               subplot(numel,2,2*k-1)
               mov_av = filter(b,a,temp_S(end,col(k))*D(:,col(k))');
%                mov_av(1:end-1) = mov_av(2:end) - mov_av(1:end-1);
%                mov_av(1:end-1) = mov_av(2:end) - mov_av(1:end-1);
               %mov_av = conv(D(:,col(i))',filter_ma,'valid');
               %plot(D(:,col(i)))
               %[corrmatirx]=[D(:,col(k)),acc_x,acc_y, acc_z];
               Rx = xcorr(D(:,col(k))',acc_x-mean(acc_x),'coeff');
               Ry = xcorr(D(:,col(k))',acc_y-mean(acc_y),'coeff');
               Rz = xcorr(D(:,col(k))',acc_z-mean(acc_z),'coeff');
               temp_corr = [temp_corr; Rx(length(D(:,col(k)))), Ry(length(D(:,col(k)))),Rz(length(D(:,col(k))))];
              
               
               %R=corrcoef(corrmatirx)
               %pause
               plot(mov_av')
               subplot(numel,2,2*k)
               [pxx,f] = periodogram(mov_av',[],4096, 125);
               
               [pks,locs] = findpeaks(pxx(prev_range(1):prev_range(3)),'MinPeakHeight',max(pxx(prev_range(1):prev_range(3)))/2);
              
               [pks_harm,locs_harm] = findpeaks(pxx(prev_harm_range(1):prev_harm_range(3)),'MinPeakHeight',max(pxx(prev_harm_range(1):prev_harm_range(3)))/2);
               [sort_pks, p_index] = sort(pks,'descend');
               [sort_h_pks, ph_index] = sort(pks_harm,'descend');
               [sort_pks]= [sort_pks, locs(p_index)+prev_range(1)-1];
               [sort_h_pks]=[sort_h_pks,locs_harm(ph_index)+prev_harm_range(1)-1];
             %  all_pks=[all_pks;sort_pks, locs+prev_range(1)];
             %  all_har_pks=[all_har_pks;sort_h_pks,locs_harm+prev_harm_range(1)];
               
               % peak and harmonic comparision for each dictionary atom
               
%                [sort_pks(:,1),I1] = sort(sort_pks(:,1),'descend');
%             temp_sort_pks = sort_pks(:,2);
%             for i = 1:size(sort_pks,1)
%                 sort_pks(i,2) = temp_sort_pks(I1(i,1),1);
%             end
            sort_pks;
            sort_h_pks;
            %[all_har_pks(:,1),I2] = sort(all_har_pks(:,1),'descend')
          
            %length(fftshift(abs(fft(D(:,1)))));
            if no==1
                 prev_f_bin=round((BPM0(1,1)*4096)/(Fs*60))+1;
            else
                prev_f_bin=round((BPM(end)*4096)/(Fs*60))+1;
            end
            heart_set=[];
            %heart_set_fbin=[];
            heart_flag=0;
            if size(sort_pks,1)==1
                comp_iter=1;
            else
                comp_iter=2;
            end
            if size(sort_h_pks,1)==1
                comp_h_iter=1;
            elseif size(sort_h_pks,1)==0
                comp_iter=0;
            else
                comp_h_iter=2;
            end
            
            for i=1:size(sort_pks,1)
                if size(sort_pks,1) == 0
                    break;
                else
                    for j=1:size(sort_h_pks,1)
                        if sort_pks(i,2)*2==sort_h_pks(j,2)-1 %|| sort_pks(i,2)*2==sort_h_pks(j,2)+1 || sort_pks(i,2)*2==sort_h_pks(j,2)
                            heart = 2047 + sort_pks(i,2);
                            [heart_list]=[heart_list,heart];
                            heart_flag=1;
                            fprintf('yes\n');
                            break;
                        else
                           heart_set=[heart_set,((sort_h_pks(j,2)-1)/2)+1-prev_f_bin];
                           %heart_set_fbin=[heart_set_fbin,(all_har_pks(j)-1)/2+1];
                        end
                    end
                    heart_set=[heart_set,sort_pks(i,2)-prev_f_bin];
                     %heart_set_fbin=[heart_set_fbin,all_pks(i)]
               end
            end
            if heart_flag==0 && size(sort_pks,1) ~= 0 && size(heart_set,1)~=0
                [heart_set_temp,ind]=sort(abs(heart_set));
                 heart = 2047+heart_set(ind(1)) + prev_f_bin;
                [heart_list]=[heart_list,heart];
            end
                
            
                %pause
            plot(f(1:upper+1),pxx(1:upper+1),'b',heart_range,zeros(1,3),'b*',prev_heart_range/60,zeros(1,3),'r*','MarkerSize',5);
            if size(sort_pks,1) ~=0
                hold on
                plot(sort_pks(:,2)*Fs/4096,sort_pks(:,1),'black*');
                plot(sort_h_pks(:,2)*Fs/4096,sort_h_pks(:,1),'g*');
                hold off
            end

            end
               
             
               
%                [pks,locs] = findpeaks(pxx(1:upper+1));
              % max_peak_loc = find(pks == max(pks));
%                [rel_pks,rel_locs] = findpeaks(pxx(prev_range(1):prev_range(3)));
               %vline(heart_range);
               %plot(-Fs/2:Fs/fft_size:(Fs/2)-(Fs/fft_size),fftshift(abs(fft(D(:,col(i)),4096))));
               %temp_fft = fftshift(abs(fft(temp_S(1,col(i))*D(:,col(i)),4096)));
               %[arranged, order] = sort(pxx,'descend');
               %maxima = [maxima;pks(max_peak_loc)'];%, max(fftshift(abs(fft(D(:,col(i)),4096))))];
               %heartbeat_temp = [heartbeat_temp; (2047 + prev_range(1)+ locs(max_peak_loc))'];
               %pause
              % heartbeat = [heartbeat;(((heartbeat_temp - (fft_size/2))/(fft_size/2))*(Fs/2)*60)];
           
            
            
            
            %length_fft = length(fftshift(abs(fft(D(:,1),4096))))
            %maxima = max(fftshift(abs(fft(D(:,1),4096))));
%             heartbeat_temp = find(fftshift(abs(fft(D(:,1),4096)))==maxima)
%             heartbeat = ((heartbeat_temp(2) - (fft_size/2))/(fft_size/2))*(Fs/2)*60
            %maxima
            %maxima=sort(maxima); 
            %temp = find(maxima == max(maxima));
          
           
            %heart = heartbeat_temp(temp);
            %heartbeat = ((heartbeat_temp - (fft_size/2))/(fft_size/2))*(Fs/2)*60;
            
            
%             if count>2
%                 BPM(end)
%             
%            % new code
%            %
% 
%            i=1;
%              while i<=size(maxima) 
%                %heartbeat=maxima(i);
%             temp = find(maxima(i));
%             heart = heartbeat_temp(temp);
%             %heartbeat = ((heartbeat_temp - (fft_size/2))/(fft_size/2))*(Fs/2)*60;
%             heartbeat = ((heart - (fft_size/2))/(fft_size/2))*(Fs/2)*60;
%           
%                if heartbeat>= BPM(end)-4 && heartbeat<=BPM(end)+4
%                     break;
%                else
%                    i
%                    i=i+1;
%                end
%             end
%             end
           %}
          %  maxima
         
              temp=abs(heart_list-prev_f_bin-2047);
              [sorted, idx]=sort(temp);
              heart=heart_list(idx(1));
            % heart=mean(heart_list);
                temp_corr
                [heartbeat] = ((heart_list(idx) - (fft_size/2))/(fft_size/2))*(Fs/2)*60
                BPM = [BPM;heartbeat(1)];
                orig_heartbeat = BPM0(count,1);
                heartbeat;
                orig_heartbeat
                %pause
                %pause 
                predicted_hb(count,1) = heartbeat(1);
                original_hb(count,1) = orig_heartbeat;
                count = count + 1;
                temp=0;
                 figure(5)
                 subplot(2,1,1)
                 plot(signal); 
                 hold on
                 plot(D*(S(end,:).*Z(end,:))','r')
                 plot(D(:,1)*(S(end,1).*Z(end,1))','g','LineWidth',2)
                 hold off
                 subplot(2,1,2)
                 hold on
                 plot(original_hb,'b*')
                 plot(predicted_hb,'r*')
                 hold off
                [Result,HR] = coherence_match26(D,Z,sig,no,HR);
      end
               
    
    
% else
%     
%     %batch mode
%     [idexi,idexj] = ind2sub(sizeIMin(1:2)-PatchSize+1,1:(sizeIMin(1)-PatchSize+1)*(sizeIMin(2)-PatchSize+1));
%     idex = [idexi',idexj'];
%     clear idexi idexj
%     X_k = IMin; %im2col(IMin,[PatchSize,PatchSize],'sliding');
%     [P,N] = size(X_k);    
%     %Sparsity Priors
%     if strcmp(InitOption,'SVD')==1
%         a0=1;
%         b0=N/8;
%     else
%         a0=1;
%         b0=1;
%     end
%     [D,S,Z,phi,alpha,Pi] = InitMatrix_Denoise(X_k,K,InitOption,IsSeparateAlpha,IMin);    
%     Pi = Pi-Pi+0.001;
%     X_k = X_k - D*S';
%     maxIt = IterPerRound(end,end);
%     for iter=1:maxIt
%         tic
%         if size(IMin,3)==1
%             Pi(1) = 1;
%         end
%         [X_k, D, Z, S] = SampleDZS(X_k, D, Z, S, Pi, alpha, phi, true, true, true);
%         Pi = SamplePi(Z,a0,b0);
%         alpha = Samplealpha(S,e0,f0,Z,alpha);
%         phi = Samplephi(X_k,c0,d0);
%         ittime=toc;
%         
%         NoiseVar(end+1) = sqrt(1/phi)*255;
%         if ReduceDictSize && iter>20
%             sumZ = sum(Z,1)';
%             if min(sumZ)==0
%                 Pidex = sumZ==0;
%                 D(:,Pidex)=[];
%                 K = size(D,2);
%                 S(:,Pidex)=[];
%                 Z(:,Pidex)=[];
%                 Pi(Pidex)=[];
%             end
%         end        
%         
% %         if iter>20
% %             %Iout    =   DenoiseOutput(D*(S.*Z)',sizeIMin,PatchSize,idex,MuX);
% %             Iout    =   DenoiseOutput_LowMemoryReq(D,S,sizeIMin,PatchSize,idex);
% %             ave.Count = ave.Count + 1;
% %             if ave.Count==1
% %                 ave.Iout = Iout;
% %             else
% %                 ave.Iout= 0.85*ave.Iout+0.15*Iout;
% %             end
% %             PSNRave = -10*log10((mean((ave.Iout(:)-IMin0(:)).^2)));
% %             PSNR(end+1) = -10*log10((mean((Iout(:)-IMin0(:)).^2)));
%             if mod(iter,10)==1
%                 disp(['iter:', num2str(iter), '    time: ', num2str(ittime)]);%'    ave_Z: ', num2str(full(mean(sum(Z,2)))),'    M:', num2str(nnz(mean(Z,1)>1/1000)), '    PSNR:',num2str(PSNR(end)),'    PSNRave:',num2str(PSNRave),'   NoiseVar:',num2str(NoiseVar(end)) ])
%                 %save([IMname,'_Denoising_',num2str(sigma),'_',num2str(iter)], 'Iout', 'D','PSNR','Pi','IMin','IMin0','phi','alpha','NoiseVar','idex','ave');
%             end
% %         end
%     end    
end
%save( [IMname,'_Denoising_',num2str(sigma)], 'Iout', 'D','S','Z','PSNR','Pi','IMin','IMin0','phi','alpha','NoiseVar','idex','ave');
%save( [IMname,'_Denoising_',num2str(sigma)], 'D','S','Z','Pi','IMin','IMin0','phi','alpha','NoiseVar');

end