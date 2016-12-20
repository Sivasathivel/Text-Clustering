%% Text Analysis for Talentica
close all; clear all;clc;
%% Get Data
[ndata, text, ~] = xlsread('/Users/sivasathivelkandasamy/Projects/Projects/Talentica DS Test/data.xlsx');
nDocs = length(ndata);
%% Construct Unique List of words
wordList = {};
for i = 1:nDocs
    wordList = cat(2,wordList,strsplit(text{i}));
end
wordList = unique(wordList);
nWords = length(wordList);
%% Construct Document Term Matrix for the docs
dtMatrix = zeros(nDocs,nWords);
for i = 1:nDocs
    for j = 1:nWords
        dtMatrix(i,j) = length(strfind(text{i},wordList{j}));
    end
end
%% Dimensionality Reduction using PCA
[coeff,~,~,~,explained] = pca(dtMatrix);
% Find the number of components required to explain 99% of the variance
i = 1;
while sum(explained(1:i)) < 99
    i = i+1;
end
nMatrix = dtMatrix * coeff(:,1:i);
%% KMeans Analysis - Determine the number of clusters
eva = evalclusters(nMatrix,'kmeans','calinskiharabasz','klist',1:10);
optimalK = eva.OptimalK;
%% K Means Clustering 
[idx,C,sumd,D]          = kmeans(nMatrix,optimalK);
%% Visualization
[coeff,~,~,~,explained] = pca(nMatrix);
pMatrix                 = nMatrix * coeff(:,1:2);

figure;plotmatrix(pMatrix);
colors = {'r','g','b','m'};

figure;
for i = 1:optimalK
    plot(pMatrix(idx==i,1),pMatrix(idx==i,2),'o','color',colors{i}); hold on;
end

hist(dtMatrix)
