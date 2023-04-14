function [fisheyeParams, estimerror] = scaramuzza(mrows, ncols, imPoints, squareSize)

    format long

%    if strcmp(medium, "air")
%        squareSize = 20;  % 'mm'
%    elseif strcmp(medium, "water")
%        squareSize = 16;  % 'mm'
%    end


    % Generate coordinates of the checkerboard corners
    worldPoints = generateCheckerboardPoints([8 8], squareSize);

    [fisheyeParams, ~, estimerror] = estimateFisheyeParameters(imPoints, worldPoints, [mrows ncols], 'EstimateAlignment', true);

    fisheyeParams = struct(fisheyeParams);
    fisheyeParams.Intrinsics = struct(fisheyeParams.Intrinsics);
    fisheyeParams.Intrinsics.UndistortMap = struct(fisheyeParams.Intrinsics.UndistortMap);

    estimerror = struct(estimerror.IntrinsicsErrors);

end