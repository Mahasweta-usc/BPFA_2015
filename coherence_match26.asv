function [Result,HR] = coherence_match26(D,Z,S,sig,no,HR)
filename = 'C:\Users\deel\Desktop\competition_data\competition\DATA_02_TYPE02.mat';
load(filename);
indices = [];
Seconds = 8;
N = 4096;
J = 0;
mse_coherence_atoms = [];
acc_X = sig(4,(no-1)*125 + 1: no*125 + (Seconds-1)*125);
acc_Y = sig(5,(no-1)*125 + 1: no*125 + (Seconds-1)*125);
acc_Z = sig(6,(no-1)*125 + 1: no*125 + (Seconds-1)*125);
z = Z(end,:); %row vector
s = S(end,:);
for i = 1:1000
    acc_net(i) = (acc_X(i)^2 + acc_Y(i)^2 + acc_Z(i)^2)^0.5;
end
accfft = abs(fft(acc_net,4096)); 
N_2 = ceil(N/2);
acc_fft = (accfft((N_2 +1):N));
for i = 1:length(z) %assuming dictionary length k = 256
    if(z(i) == 1)
        d = D(:,i); %column vector
        dfft = (abs(fft(d,4096,2)));
        d_fft = (dfft(N_2 + 1:N));
        indices = [indices,i];
        [coherence,w] = mscohere(d',acc_net,[],[],4096,125);
        count = 0;
        summ = 0;
        for j = 1:length(d_fft)
            main_freq = max(acc_fft);
            if(acc_fft(j) >= 0.65*main_freq) %
                summ = summ + coherence(j)^2;
                count = count + 1;
            end
        end
        mse = (summ/count)^0.5;
            mse_coherence_atoms = [mse_coherence_atoms,mse];
    end
end
mse_coherence_atoms
indices_coh = [];
count = 1;
sort_mse_coherence_atoms = sort(mse_coherence_atoms,'ascend');
min_coh = sort_mse_coherence_atoms(1);
if(min_coh <= 0.25*max(mse_coherence_atoms))
    count = count + 1;
    min_coh = sort_mse_coherence_atoms(count);
end
for i = 1:length(mse_coherence_atoms)
    if(mse_coherence_atoms(i) <= 1.25*min_coh)
        indices_coh = [indices_coh,indices(i)];
    end
end
D_net = [];
S_net = [];
for j = 1:length(indices_coh)
    D_net = horzcat(D_net,D(:,indices_coh(j)));
    S_net = [S_net,s(indices_coh(j))];
end
Result = D_net*(S_net');
Result = full(Result);
Resultfft = abs(fft(Result',4096));
Result_fft = Resultfft(2049:4096);
frequencies = [];
indices2 = [];
trend = golayfilt_trend(sig,no);
HR = harmonics_track(Result_fft,HR);
HR
end
       
