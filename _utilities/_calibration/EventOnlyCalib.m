%% Code to calbirate scale and bias of an event camera using pure events 
% No frame is used in this event only calibration method
% Input 1: sumE (sum of event count between each frame timestamp)  
% Input 2: sumP(sum of polarity between each frame timestamp)
% Output: save calibration parameter scale and bias in .csv files
clear
close all

%% Parameter initialization
% Dataset to be calibrated
% Option 1: dynamic
% Option 2: box_translation
% Option 3: office

data_directory = "F:\Honours\spatial_compression\resolution_[100-100]\dvs_vpr_2020-04-21-17-03-03.txt";

% Define frame dimensions
data_directory_split = split(data_directory, '\');
resolution_string = data_directory_split{4};
resolution_string_split = split(resolution_string, '-');
resolution = str2double(resolution_string_split{2}(1:end-1));

% start_idx and end_idx define how many events are used in calibration
start_idx = 1;
end_idx = 50000;

% Default contrast threshold
ct_default = 0.1;
% Frame height
height = resolution;
% Frame width
width = resolution;
% Linear regression coefficient 1
m = zeros(height,width);
% Linear regression coefficient 2
d = zeros(height,width);
% Root mean square error
r = zeros(height,width);

%% Load sum of event count and sum of polarity
count = 1;
% Sum of event count and sum of polarity
sumE = zeros(height,width,end_idx - start_idx + 1);
sumP = zeros(height,width,end_idx - start_idx + 1);
for i=start_idx:end_idx
    idx = num2str(i);
	sumE(:,:,count) = load(['./data/' + dataset + '/sumE/data_event_' + idx + '.txt'],'%s','delimiter',',');
    sumP(:,:,count) = load(['./data/' + dataset + '/sumP/data_polarity_' + idx + '.txt'],'%s','delimiter',',');
    count = count + 1;
end



%% Refer to Section 4.3 in the paper
% https://ssl.linklings.net/conferences/acra/acra2019_proceedings/views/includes/files/pap135s1-file1.pdf
for i = 1:height
    for j = 1:width
        P = reshape(sumP(i,j,:),[1,end_idx-start_idx+1]);
        E = reshape(sumE(i,j,:),[1,end_idx-start_idx+1]);
        A = [E', ones(size(E))'];
        if (rcond(A'*A) < 1e-14) 
            r(i,j) = NaN;
        else
            coef = (A'*A)\A'*P';
            m(i,j) = coef(1);
            d(i,j) = coef(2);
            r(i,j) = sqrt(mean((P - (m(i,j) * E + d(i,j))).^2));
        end
    end
end

% Compute scale = median(r) / r, refer to Equation (11)
scale = median(ct_default * r(~isnan(r))) ./ r;
% Compute bias, refer to Equation (12)
bias = - scale .* m;
% Replaces NaN or unrealistically large scale values
replace_mask = (abs(scale) > 10 * ct_default) | isnan(r);
scale(replace_mask) = ct_default;
bias(replace_mask) = 0;

%% Save calibration parameters
folderCalibParam = ['./results/' + dataset + '/EventOnlyCalib/'];
if ~exist(folderCalibParam, 'dir')
    mkdir(folderCalibParam);
end

csvwrite(sprintf([folderCalibParam + '/scale.csv']),scale);
csvwrite(sprintf([folderCalibParam + '/bias.csv']),bias);