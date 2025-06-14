clear; clc; close all;

%% Setup and Load Data
recordLocation = "C:\Users\squddus\Documents\Radar Fall Data Generation (Standardized Dataset)\Original Data\Falls\forwardfallJN";

% Load the 4D radar data cube
load(fullfile(recordLocation, "4DdataCube.mat"), "radarDataCube");
pause(2);

% Load recording parameters
temp = load(fullfile(recordLocation, "iqData_RecordingParameters.mat"));
dcaRecordingParams = temp.RecordingParameters;

% Extract radar parameters
Fs = dcaRecordingParams.ADCSampleRate * 1e3;  % Hz
sweepSlope = dcaRecordingParams.SweepSlope * 1e12;
nr = dcaRecordingParams.SamplesPerChirp;
fc = dcaRecordingParams.CenterFrequency * 1e9;  % Hz
tpulse = 2 * dcaRecordingParams.ChirpCycleTime * 1e-6;
PRF = 1 / tpulse;
nrx = dcaRecordingParams.NumReceivers;
numChirps = dcaRecordingParams.NumChirps;

c = 3e8; % Speed of light

%% Range Axis Setup
rangeAxis = (0:nr-1) * c * Fs / (2 * sweepSlope * nr);  % accurate range vector
maxRange = 5;  % meters
rangeIndices = find(rangeAxis <= maxRange);  % keep bins up to 5m
rangeAxis = rangeAxis(rangeIndices);

%% Preallocate Range-Time Matrix
rxIdx = 1;
[numSamples, ~, numChirps, numFrames] = size(radarDataCube);
rangeTimeMap = [];

for frameIdx = 1:numFrames
    % Get data for receiver
    data = squeeze(radarDataCube(:, rxIdx, :, frameIdx));  % [samples x chirps]

    % Remove DC/clutter
    data = data - mean(data, 2);

    % Perform Range FFT (on fast time axis)
    rangeFFT = fft(data, nr, 1);  % size: [range_bins x chirps]
    rangeFFT = rangeFFT(rangeIndices, :);  % keep only bins up to 5m

    % Average across chirps to reduce noise
    rangeProfile = mean(abs(rangeFFT), 2);  % [range_bins x 1]

    % Store each frame's range profile
    rangeTimeMap(:, frameIdx) = rangeProfile;  % grows to [range_bins x numFrames]
end

%% Plot Range-Time Spectrogram
t = (0:numFrames-1) * tpulse;  % time axis

figure;
imagesc(t, rangeAxis, 20*log10(abs(rangeTimeMap) + eps));
axis xy;
xlabel('Time (s)');
ylabel('Range (m)');
title('Range-Time Spectrogram');
colormap('gray');  % match the dataset's grayscale style
colorbar;
