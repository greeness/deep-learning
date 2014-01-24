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

for poolRow= 1:floor(convolvedDim / poolDim)
    for poolCol = 1:floor(convolvedDim / poolDim)
        rowIdxRange = 1 + (poolRow-1) * poolDim : poolRow * poolDim;
        colIdxRange = 1 + (poolCol-1) * poolDim : poolCol * poolDim;
        block = convolvedFeatures(:, :, rowIdxRange, colIdxRange);
        blockMean = mean(block(:));
        pooledFeatures(:, :, poolRow, poolCol) = blockMean;
    end
end

end

