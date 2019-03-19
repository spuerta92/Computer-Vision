% Algorithm details with SSD.
% Let SampleImage contain the image we are sampling from and let Image be
% the mostly empty image that we want to fill in (if synthesizing from 
% scratch, it should contain a 3-by-3 seed in the center randomly taken
% from SampleImage, for constrained synthesis it should contain all the 
% known pixels). WindowSize, the size of the neighborhood window, is the
% only user-settable parameter. The main portion of the algorithm is 
% presented below.
function GrowImage(SampleImage,Image)
    while Image not filled do
        progress = 0;
        PixelList = GetUnfilledNeighbors(Image);
        for i = 1:len(PixelList)
            Template = GetNeighborhoodWindow(Pixel);
            BestMatches = FindMatches(Template, SampleImage);
            BestMatch = RandomPick(BestMatches);
            %Pixel.value = BestMatch.value
        end
    end
    return Image
end

