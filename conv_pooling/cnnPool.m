function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, ...
                       numImages, ...
                       floor(convolvedDim / poolDim), ...
                       floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://deeplearning.stanford.edu/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

% poolDim is 19x19, 
%             #feature x #images x 3 x 3
% pooledFeatures : 400 x 8 x 3 x 3

fprintf('Total features %d\n', numFeatures);
for feature = 1:numFeatures
    if mod(feature, 10) == 0
        fprintf('Pooling feature %3d\n', feature);
    end
    for image = 1:numImages
        for poolRow= 1:floor(convolvedDim / poolDim)
            startRow = 1 + (poolRow-1) * poolDim;
            endRow = poolRow * poolDim;
            rowIdxRange = startRow : endRow;
            for poolCol = 1:floor(convolvedDim / poolDim)
                startCol = 1 + (poolCol-1) * poolDim;
                endCol = poolCol * poolDim;
                colIdxRange = startCol : endCol;

                block = convolvedFeatures(feature, image, rowIdxRange, colIdxRange);
                blockMean = mean(block(:));

                %fprintf('r%d->%d, c%d->%d, block mean %f\n', startRow, endRow, startCol, endCol, blockMean);
                pooledFeatures(feature, image, poolRow, poolCol) = blockMean;
            end
        end
    end
end

end


%expectedMatrix = [mean(mean(testMatrix(1:4, 1:4))) mean(mean(testMatrix(1:4, 5:8))); ...
%size                  mean(mean(testMatrix(5:8, 1:4))) mean(mean(testMatrix(5:8, 5:8))); ];

