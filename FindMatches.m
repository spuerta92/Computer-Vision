% Function GetUnfilledNeighbors() returns a list of all unfilled pixels that
% have filled pixels as their neighbors (the image is subtracted from its 
% morphological dilation). The list is randomly permuted and then sorted by 
% decreasing number of filled neighbor pixels. GetNeigborhoodWindow() 
% returns a window of size WindowSize around a given pixel.
% RandomPick() picks an element randomly from the list. FindMatches() is 
% as follows:

function FindMatches(Template,SampleImage)
    % ValidMask = 1s where Template is filled, 0s otherwise
    TotWeight = sum i,j ValidMask(i,j)
    for i,j do
        for ii,jj do
        dist = (Template(ii,jj)-SampleImage(i-ii,j-jj))^2
        SSD(i,j) = SSD(i,j) + dist*ValidMask(ii,jj)
        end
        SSD(i,j) = SSD(i,j) / TotWeight
    end
    PixelList = all pixels (i,j) where SSD(i,j) <= min(SSD)*(1+ErrThreshold)
    return PixelList
end
% In our implementation the constant were set as follows: 
% ErrThreshold = 0.1. Pixel values are in the range of 0 to 1.