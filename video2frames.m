
function video2frames(parentDir, varargin)
% video2frames(parentDir, Name,Value)
% Scans all direct subfolders of parentDir. In each subfolder, looks for
% "scene_camera.mp4", creates an "image_2" folder, and exports frames as
% PNG: 000001.png, 000002.png, ...
%
% Name-Value args:
%   'SkipExisting' (true/false, default true)  -> skip subfolders that already have PNGs
%   'UseFFmpegFirst' (true/false, default false) -> try ffmpeg before VideoReader

    ip = inputParser;
    addParameter(ip, 'SkipExisting', true, @(x)islogical(x) && isscalar(x));
    addParameter(ip, 'UseFFmpegFirst', false, @(x)islogical(x) && isscalar(x));
    parse(ip, varargin{:});
    opts = ip.Results;

    if ~isfolder(parentDir)
        error('Parent directory not found: %s', parentDir);
    end

    items = dir(parentDir);
    items = items([items.isdir]);               % only directories
    names = {items.name};
    mask  = ~ismember(names, {'.','..'});
    items = items(mask);

    for k = 1:numel(items)
        subFolder = fullfile(parentDir, items(k).name);
        videoFile = fullfile(subFolder, 'scene_camera.mp4');
        outDir    = fullfile(subFolder, 'image_2');

        if ~isfile(videoFile)
            fprintf('No scene_camera.mp4 in: %s\n', subFolder);
            continue;
        end

        % If output already exists and has PNGs, optionally skip
        if opts.SkipExisting && isfolder(outDir) && ~isempty(dir(fullfile(outDir, '*.png')))
            fprintf('Skipping (already has PNGs): %s\n', subFolder);
            continue;
        end

        if ~exist(outDir, 'dir'); mkdir(outDir); end

        ok = false;
        if opts.UseFFmpegFirst
            ok = try_ffmpeg(videoFile, outDir);
            if ~ok
                ok = try_videoreader(videoFile, outDir);
            end
        else
            ok = try_videoreader(videoFile, outDir);
            if ~ok
                ok = try_ffmpeg(videoFile, outDir);
            end
        end

        if ~ok
            fprintf(2, 'FAILED: %s\n', videoFile);
        end
    end
end

function ok = try_videoreader(videoFile, outDir)
    ok = false;
    try
        v = VideoReader(videoFile);
        frameCount = 0;
        while hasFrame(v)
            frameCount = frameCount + 1;
            im = readFrame(v);
            fn = fullfile(outDir, sprintf('%06d.png', frameCount));
            imwrite(im, fn);
        end
        fprintf('Extracted %d frames with VideoReader: %s\n', frameCount, videoFile);
        ok = frameCount > 0;
    catch ME
        fprintf(2, 'VideoReader failed for %s\nReason: %s\n', videoFile, ME.message);
    end
end

function ok = try_ffmpeg(videoFile, outDir)
    ok = false;
    % Build ffmpeg command. -vsync 0 avoids duplicate/dropped-frame renumbering.
    % -start_number 1 ensures 000001.png, 000002.png, ...
    outPattern = fullfile(outDir, '%06d.png');
    cmd = sprintf('ffmpeg -y -hide_banner -loglevel error -vsync 0 -i "%s" -start_number 1 "%s"', videoFile, outPattern);
    status = system(cmd);

    if status == 0
        n = numel(dir(fullfile(outDir, '*.png')));
        fprintf('Extracted %d frames with ffmpeg: %s\n', n, videoFile);
        ok = n > 0;
    else
        fprintf(2, 'ffmpeg failed for %s (is ffmpeg installed and on PATH?)\n', videoFile);
    end
end
