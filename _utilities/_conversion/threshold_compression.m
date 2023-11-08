%% Add file paths
clearvars
clc

linux = 0;

if linux == 1
    % do something
    data_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/';  
    save_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/';
else
    data_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/';  
    save_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/';
end

%---- Parameters ----%
threshold_array = [44, -32];

% CHANGE AS NEEDED
rows = 100;
cols = 100;
dt = 0.5; % down time before resting the accumulator
print_interval = 1; % seconds
end_time = 300;


%---- sunset1 and sunset2 ----%
% We will only look at these two files for threshold and resolution testing
filenames = ["dvs_vpr_2020-04-21-17-03-03.txt";
             "dvs_vpr_2020-04-22-17-24-21.txt"];
             
for thresh_index = 1:size(threshold_array, 1)
    pos_threshold = threshold_array(thresh_index, 1);
    neg_threshold = threshold_array(thresh_index, 2);
    fprintf('Positive Threshold: %d \t Negative Threshold: %d \n', pos_threshold, neg_threshold);

    %---- Loop through reading and saving data ----$
    for i = 1:length(filenames) 
        % Open the file
        filename = filenames(i);
        fid = fopen(data_dir + filename, 'r');
        savename = erase(filename, ".txt");

        % Setup up loop variables
        counter = 1; % counter for while loop
        element_counter = 1;
        t_array = [];
        x_array = [];
        y_array = [];
        pol_array = [];

        % Arrays for data compression
        M_accumulator = zeros(rows, cols);
        M_prev_time = zeros(rows, cols);

        % variables for printing timer
        prev_interval_time = 0;
        interval_time = 0;

        while true

            tline = fgetl(fid);

            if ischar(tline)
                data = split(tline);
                format long;
                t = str2double(data{1});
                x = str2double(data{2}) + 1;
                y = str2double(data{3}) + 1;
                pol = str2double(data{4});

                if counter == 1
                    t_0 = t;
                    counter = counter + 1;
                end

                % Assign pol accordingly
                if pol == 0
                    pol_signed = -1;
                elseif pol == 1
                    pol_signed = 1;
                else
                    error("Polarity Error");
                end

                % add event to the accumulator
                M_accumulator(y, x) = M_accumulator(y, x) + pol_signed;
                
                % release the event if it exceeds the threshold
                if (M_accumulator(y, x) >= pos_threshold) || (M_accumulator(y, x) <= neg_threshold)
                    % fprintf("Threshold Satsified \t Count: %d \n", M_accumulator(y, x));
                    if (M_accumulator(y, x) >= pos_threshold)
                        pol_store = 1;
                    elseif (M_accumulator(y, x) <= neg_threshold)
                        pol_store = 0;
                    end

                    t_array(element_counter) = t;
                    x_array(element_counter) = x - 1;
                    y_array(element_counter) = y - 1;
                    pol_array(element_counter) = pol_store;
                    element_counter = element_counter + 1;
                    M_accumulator(y, x) = 0;
                end

                % check the timing since previous events and clear if
                % necessary
                current_time = t-t_0;
                M_prev_time(y,x) = current_time;
                % create mask of pixels that haven't had events
                M_mask = (current_time - M_prev_time) > dt;
                M_accumulator(M_mask) = 0; 

                % print time every second to show that it is progressing
                interval_time = (t - t_0) - prev_interval_time;
                if interval_time > print_interval
                    prev_interval_time = prev_interval_time + interval_time;
                    interval_time = 0;
                    fprintf("Time: \t %.2f\n", t-t_0);
                end

                % check the timing, and save file if necessary
                if (t-t_0) >= end_time
                    break
                end

            else 
                break;
            end
        end

        data = [t_array; x_array; y_array; pol_array]';
        save_dir_batch = save_dir + sprintf("pos_%d_neg_%d_2/", pos_threshold, neg_threshold);
        if ~isfolder(save_dir_batch)
            mkdir(save_dir_batch);
        end
        save(save_dir_batch+savename+".mat", "data");
        fprintf("%s complete \n", filename);
        fclose(fid);
    end
end
