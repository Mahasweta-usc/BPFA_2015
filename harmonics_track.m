function[HR] = harmonics_track(Result_fft,HR)
range = (ceil(((HR-10)*(4096/(125*60)) + 1))):1:ceil(((HR+15)*(4096/(125*60)) + 1));
Result_fft_range = Result_fft(ceil(((HR-10)*(4096/(125*60)) + 1)):1:ceil(((HR+15)*(4096/(125*60)) + 1)));
max_Result_fft_range = max(Result_fft_range);
indices2 = [];
frequencies = [];
for k = 1:length(range)
    if(Result_fft_range(k) >= 0.7*max_Result_fft_range)
        indices2 = [indices2,range(k)];
        frequencies = [frequencies,Result_fft_range(k)];
    end
end
[sortedValues,sortIndex] = sort((frequencies),'descend');
% if(length(sortedValues)>= 3)
% max_3 = sortedValues(1:3);
% else
%         max_3 = sortedValues;
% end
indices3=[];
for m = 1:1:length(sortedValues)
    for p = 1:1:length(frequencies)
        if(frequencies(p) == sortedValues(m))
            indices3 = [indices3,indices2(m)];
        end
    end
end
range2 = (ceil(((HR-4)*(4096/(125*60)) + 1))):1:ceil(((HR+8)*(4096/(125*60)) + 1));
Result_fft_range2 = Result_fft(ceil(2*((HR-4)*(4096/(125*60)) + 1)):1:ceil(2*((HR+8)*(4096/(125*60)) + 1)));
max_Result_fft_range2 = max(Result_fft_range);
indices4 = [];
frequencies2 = [];
for k = 1:length(range2)
    if(Result_fft_range2(k) >= 0.7*max_Result_fft_range2)
        indices4 = [indices4,range2(k)];
        frequencies2 = [frequencies2,Result_fft_range2(k)];
    end
end
[sortedValues2,sortIndex2] = sort((frequencies2),'descend');
% if(length(sortedValues2)>= 5)
% max_3_2 = sortedValues2(1:5);
% else
%     if(length(max_3) <= length(sortedValues2))
%         max_3_2 = sortedValues2(1:length(max_3));
%     else
%         min_dim = min(length(max_3,sortedValues2));
%         max_3_2 = sortedValues2(1:min_dim);
%         max_3 = sortedValues(1:min_dim);
%     end
%         
% end
indices5=[];
for m = 1:1:length(sortedValues2)
    for p = 1:1:length(frequencies2)
        if(sortedValues2(m) == frequencies2(p))
            indices5 = [indices5,indices4(m)];
        end
    end
end
x = length(indices5);
y = length(indices3);
A = [x,y,5];
min_dim = min(A);
indices3_new = indices3(1:min_dim);
indices5_new = indices5(1:min_dim);
% indices5 contains frequencies corresponding to amplitudes in descending
% order
fundamentals = (indices3_new -1)*(125/4096)*60;
first_harmonics = (indices5_new -1)*(125/4096)*60;
fundamentals = sort(fundamentals);
first_harmonics = sort(first_harmonics);
diff1 = [];
for i = 1:length(fundamentals)
    diff1 = [diff1,abs(2*fundamentals(i) - first_harmonics(i))];
end
[min_diff,I] = min(diff1);
if((min_diff <= 0.025*fundamentals(I))&&(abs(HR-fundamentals(I))<=10))
    HR = fundamentals(I);
else
range3 = (ceil(HR-4)*(4096/(125*60)) + 1):ceil((HR+8)*(4096/(125*60)) + 1);
Result_fft_range3 = Result_fft(ceil(HR-4)*(4096/(125*60)) + 1):ceil((HR+8)*(4096/(125*60)) + 1);
max_Result_fft_range3 = max(Result_fft_range3);
indices6 = [];
frequencies3 = [];
for k = 1:length(range3)
    if(Result_fft_range3(k) >= 0.7*max_Result_fft_range3)
        indices6 = [indices6,range3(k)];
        frequencies3 = [frequencies3,Result_fft_range3(k)];
    end
end
[sortedValues3,sortIndex3] = sort((frequencies3),'descend');
% if(length(sortedValues)>= 3)
% max_3 = sortedValues(1:3);
% else
%         max_3 = sortedValues;
% end
indices7=[];
for m = 1:1:length(sortedValues3)
    for p = 1:1:length(frequencies3)
        if(frequencies3(p) == sortedValues3(m))
            indices7 = [indices7,indices6(m)];
        end
    end
end
HR = sum((indices7-1)*(125/4096)*60)/length(indices7);
end
end
