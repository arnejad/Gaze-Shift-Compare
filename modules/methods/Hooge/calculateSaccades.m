

% Check if arguments were passed
if ~isdeployed
    args = getenv('MATLAB_ARGS'); % Read argument from environment variable
    splitArgs = strsplit(args, ' ');

    directoryPath = splitArgs{1};
    windowSize = str2double(splitArgs{2});
    lmda= str2double(splitArgs{3});

    fprintf('Directory: %s\n', directoryPath);
    fprintf('Param1: %.2f\n', windowSize);
    fprintf('Param2: %.2f\n', lmda);
else
    error('This script must be run with an argument.');
end


% directoryPath = "/home/ash/projects/Wild-Saccade-Detection-Comparison/degs_cached";
% windowSize = 5000;
% lmda = 1.5;
csvFiles = dir(fullfile(directoryPath, '*.csv'));

resDir = append(directoryPath, '/', 'results');
if ~exist(resDir, 'dir')
       mkdir(resDir)
end

for i = 1:length(csvFiles)
        % Get full file path
        filePath = fullfile(directoryPath, csvFiles(i).name);
        
        % Read CSV file into a matrix
        data = readmatrix(filePath);

        preds = ones(size(data,1), 1);

        res = fixdetectmovingwindow(data(:,2), data(:,3), data(:,1), windowSize, lmda);
        
        for fix = 1:2:size(res,1)
            start = res(fix);
            ending= res(fix+1);
            start_idx = find(data(:,1)==start);
            end_idx = find(data(:,1)==ending);
            preds(start_idx: end_idx) = 0;
        end
        fprintf('Completed: %s\n', csvFiles(i).name);
        writematrix(preds, append(resDir, '/',csvFiles(i).name))
end