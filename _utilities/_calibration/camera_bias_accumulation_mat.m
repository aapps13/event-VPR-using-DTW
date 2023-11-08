%% Setup data hierarchy
clearvars 
clc

linux = 0;
if linux
    % data_dir = ...
    % save_dir = ...
else
    % USB
    % data_dir = 'G:\Honours\spatial_compression\resolution_[100-100]\thresholded\pos_20_neg_-15\';
    % save_dir = 'G:\Honours\spatial_compression\resolution_[100-100]\thresholded\pos_20_neg_-15\';
    % HARDRIVE
    data_dir = 'F:\Data\Output_Data\full_dataset\spatial_compression\resolution_[100-100]\pos_44_neg_-32\';
    save_dir = 'F:\Data\Output_Data\full_dataset\spatial_compression\resolution_[100-100]\pos_44_neg_-32\';
end

filenames = ["dvs_vpr_2020-04-21-17-03-03.mat";
             "dvs_vpr_2020-04-22-17-24-21.mat"];

%---- Parameters ----%
resolution = 100;

%---- Variables ----#
for i = 1:length(filenames)
    % setup loop variables
    iteration = 0;
    prev_interval_time_print = 0;
    accumulation_cell = cell(resolution, resolution);

    % load data
    file = filenames(i);
    load(data_dir+file);
    file = erase(file, '.mat');

    for j = 1:size(data,1)
        format long
        t = data(j, 1);
        x = data(j, 2);
        y = data(j, 3);
        pol = data(j, 4);

        % fprintf('%d \t %d \t %d \t %d\n', t, x, y, pol);

        if iteration == 0 
            t_0 = t;
            iteration = iteration + 1; 
        end

        if pol == 1
            pol_signed = 1;
        else % pol == 0 or pol == -1
            pol_signed = -1; 
        end
   
        cell_index = size(accumulation_cell{y+1,x+1},2)+1;

        % First entry, add event polarity
        if cell_index == 1
            accumulation_cell{y+1,x+1}(1:2,cell_index) = [pol_signed; t]; 
        
        % For each measurement, add the current polarity to the
        % previous accumuled value
        else
            new_val = accumulation_cell{y+1,x+1}(1,cell_index-1) + pol_signed;
            accumulation_cell{y+1,x+1}(1:2, cell_index) = [new_val; t]; 
        end


        % print time every second to show that it is progressing
        interval_time_print = (t - t_0) - prev_interval_time_print;
        if interval_time_print > 1
            prev_interval_time_print = prev_interval_time_print + interval_time_print;
            interval_time_print = 0;
            fprintf("Time: \t %.2f\n", t-t_0);
        end
    end
    
    % Close the test file

    % Save the accumulation cell
    save_filename = save_dir + file + "_bias_integration.mat";
    save(save_filename, "accumulation_cell", "-mat");
    fprintf("%s complete \n", file);
     pause(0.01);
end

