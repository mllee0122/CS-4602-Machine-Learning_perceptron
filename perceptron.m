%CS4602 Introduction to Machine Learning
%Homework3_20181025
%107064522
%Chapter4_Computer Assignment

clc;
clear;

%%%%% 4.2 %%%%%
fprintf('Computer Assignment 4.2\n\n')
%Reading the dataset from hw3_dataset.txt
[ex, val] = textread('hw3_dataset.txt', '%s %s');  %#ok<DTXTRD>
data = ones(20, 6);
for i=1:20
    for j=1:5
        data(i, j+1) = str2double(val{i}(j));
    end
end

%Using c for storing the labels
c = zeros(20, 1);
data_sum = sum(data, 2); %sum each row
for i = 1:20
    if data_sum(i, 1) >= 4
        c(i, 1) = 1;
    else
        c(i, 1) = 0;
    end
end

%Setting the learning rate
eta = [0.2 0.4 0.6 0.8];
epoch = zeros(4, 1);

%perceptron learning
for n = 1:4
    %Setting the initial weights
    w =  [0.2 0.2 0.2 0.2 0.2 0.2];
    %h(x) denotes the class returned by the classifier
    h = zeros(20, 1);
    err = zeros(20, 1); %Record c(x)-h(x)
    classifier = zeros(20, 6);
    
    epochs = 0; %number of example-presentations
    hold = 1; 

    while hold
        for i = 1:20
            if sum(data(i, :).*w, 2) > 1e-10
                h(i, 1) = 1;
            else
                h(i, 1) = 0;
            end
            err(i, 1) = c(i, 1)-h(i, 1);
            w = w+eta(n)*err(i, 1)*data(i,:);
            %Recording new classifier
            for j = 1:6
                classifier(i, j) = w(j);
            end
        end
        %Check if all of the data are correctly classified
        if sum(err ~= zeros(20, 1)) ~= 0
            epochs = epochs+1;
        else
            hold = 0;
        end
    end
    epoch(n, 1) = epochs;
    fprintf('For learning rate = %.1f :\n', eta(n))
    fprintf('\t%d epochs Done! \n', epochs)
    fprintf('\tFinal classifier : [w0, w1, w2, w3, w4, w5] = [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n\n', classifier(n, :))
end
figure
plot(eta, epoch, 'o');

%%%%% 4.3 %%%%%
fprintf('Computer Assignment 4.3\n\n')
N = [1 5 10 15 20];

sum_epoch = zeros(length(N), 1);
for n = 1:length(N)
    for k = 1:1000
        new_data = zeros(20, 6+N(n));
        new_classifier = zeros(20, 6+N(n));
        new_w = 0.2*ones(1, 6+N(n));
        epochs = 0;
        hold = 1;

        for i = 1:20
            temp = round(unifrnd(0,1,1,N(n)));
            new_data(i, :) = [data(i, :) temp];
        end

        while hold
            for i = 1:20
                if sum(new_data(i, :).*new_w, 2) > 1e-10
                    h(i, 1) = 1;
                else
                    h(i, 1) = 0;
                end
                err(i, 1) = c(i, 1)-h(i, 1);  %Using the c of previous problem
                new_w = new_w+0.2*err(i, 1)*new_data(i,:);
                %Recording new classifier
                for j = 1:6+N(n)
                    new_classifier(i, j) = new_w(j);
                end
            end
            %Check if all of the data are correctly classified
            if sum(err ~= zeros(20, 1)) ~= 0
                epochs = epochs+1;
            else
                hold = 0;
            end
        end
        sum_epoch(n, 1) = sum_epoch(n, 1)+epochs;
    end
end

for i = 1:length(N)
    fprintf('With %d attributes added\n', N(i))
    fprintf('\tAverage epochs : %.3f\n\n', sum_epoch(i, 1)/1000)
end

figure
plot(N, sum_epoch(:, 1)/1000, 'o');


