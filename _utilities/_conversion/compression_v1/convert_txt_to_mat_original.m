%% Add file paths
clearvars

%---- Parameters ----%
threshold = 750;
resolution = [2,2];
dt = 0.25;
rows = 260;
cols = 346;



%---- Setup up the files data structure ----%
save_string = sprintf("resolution_[%d-%d]_threshold_%d", resolution(1), resolution(2), threshold);

mat_dir = 'F:\Data\Brisbane_Event_Data';
% save_dir = 'F:\Data\Output_Data\full_dataset\resolution_[2-2]_threshold_750\';
save_dir = 'F:\Data\Output_Data\full_dataset\' + save_string + '\';

chdir(mat_dir)

filenames = [%"dvs_vpr_2020-04-27-18-13-29.txt";
             "dvs_vpr_2020-04-21-17-03-03.txt";
             "dvs_vpr_2020-04-22-17-24-21.txt";
             "dvs_vpr_2020-04-24-15-12-03.txt";
             "dvs_vpr_2020-04-28-09-14-11.txt";
             "dvs_vpr_2020-04-29-06-20-23.txt"];

hotpixel_filenames = [%"dvs_vpr_2020-04-27-18-13-29_hot_pixels.txt";
                      "dvs_vpr_2020-04-21-17-03-03_hot_pixels.txt";
                      "dvs_vpr_2020-04-22-17-24-21_hot_pixels.txt";
                      "dvs_vpr_2020-04-24-15-12-03_hot_pixels.txt";
                      "dvs_vpr_2020-04-28-09-14-11_hot_pixels.txt";
                      "dvs_vpr_2020-04-29-06-20-23_hot_pixels.txt"];

filenames_cell = cell(length(filenames), 1);

for i = 1:length(filenames)
    filenames_cell{i} = [filenames(i), hotpixel_filenames(i)];
end


%---- Loop through reading and saving data ----$

for i = 1:length(filenames) 
    filename = filenames_cell{i}(1);
    fid = fopen(filename);
    savename = erase(filename, ".txt");

    hotpixels = table2array(readtable(filenames_cell{i}(2)));
     
    tline = fgetl(fid); % first line of .txt file (doesn't have useful information)

    counter = 1; % counter for while loop
    element_counter = 1;
    t_array = [];
    x_array = [];
    y_array = [];
    pol_array = [];
    
    % Arrays for data compression
    M_accumulator = zeros(resolution);
    % M_prev_time = zeros(resolution);

    y_compressed_indexes = linspace(0, rows, resolution(1) + 1);
    x_compressed_indexes = linspace(0, cols, resolution(2) + 1);

    % variables for printing timer
    prev_interval_time_print = 0;
    interval_time_print = 0;
    
    while true

        tline = fgetl(fid);

        if ischar(tline)
            data = split(tline);
            format long;
            t = str2double(data{1});
            x = str2double(data{2});
            y = str2double(data{3});
            pol = str2double(data{4});

            if counter == 1
                t_0 = t;
            end

            %---- Save if not a hot pixel ----%
            pixel_coords = [x, y];
            is_hot_pixel = ismember(pixel_coords, hotpixels, 'rows');

            if ~(is_hot_pixel)
                % Assign pol accordingly
                if pol == 0
                    pol_signed = -1;
                elseif pol == 1
                    pol_signed = 1;
                else
                    error("Polarity Error");
                end

                % find coords in the down sampled resolution
                x_ind = find(x_compressed_indexes <= x, 1, 'last');% - 1;
                y_ind = find(y_compressed_indexes <= y, 1, 'last'); %- 1;

                % add to the accumulator
                M_accumulator(y_ind, x_ind) = M_accumulator(y_ind, x_ind) + pol_signed;
                % M_prev_time(y_ind, x_ind) = t-t_0;

                % release the event if it exceeds the threshold
                if abs(M_accumulator(y_ind, x_ind)) >= threshold
                    if M_accumulator(y_ind, x_ind) < 0
                        pol_store = -1;
                    elseif M_accumulator(y_ind, x_ind) > 0
                        pol_store = 1;
                    end

                    t_array(element_counter) = t-t_0;
                    x_array(element_counter) = x_ind;
                    y_array(element_counter) = y_ind;
                    pol_array(element_counter) = pol_store;
                    element_counter = element_counter + 1;
                    M_accumulator(y_ind, x_ind) = 0;
                end
            end

            counter = counter + 1;

            % print time every second to show that it is progressing
            interval_time_print = (t - t_0) - prev_interval_time_print;
            if interval_time_print > 1
                prev_interval_time_print = prev_interval_time_print + interval_time_print;
                interval_time_print = 0;
                fprintf("Time: \t %.2f\n", t-t_0);
            end

            % check the timing, and save file if necessary



        else 
            break;
        end
    end

    data = [t_array; x_array; y_array; pol_array]';
    
    save(save_dir+savename+".mat", "data");
    fprintf("%s complete \n", filename);
    fclose(fid);
end

