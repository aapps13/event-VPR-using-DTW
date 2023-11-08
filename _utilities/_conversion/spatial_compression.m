
%% Setup
clearvars

%---- Parameters ----%
resolution = [100,100];
rows = 260;
cols = 346;

linux = 0;

%---- Setup up the file structure ----%
save_folder_string = sprintf("resolution_[%d-%d]", resolution(1), resolution(2));

if linux
    complete_data_dir = "/media/aapps/Elements/Data/Brisbane_Event_Data/";
    save_dir = "/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/" + save_folder_string;
else
    complete_data_dir = 'F:\Data\Brisbane_Event_Data\\';
    save_dir = 'F:\Data\Output_Data\full_dataset\spatial_compression\' + save_folder_string;
end

chdir(complete_data_dir) % change to the directory that has all the complete data files
% 
% % check if the save directory exists, and create it if needed
% if ~isfolder(save_dir)
%     mkdir(save_dir);
% end

% check if the save directory exists, and create it if needed
if ~isfolder(save_dir)
    mkdir(save_dir);
end

save_dir = save_dir + '/'; % add slash so files are saved in the directory

%---- Combine filenames into a cell array for processing ----%
filenames = ["dvs_vpr_2020-04-21-17-03-03.txt";
             "dvs_vpr_2020-04-22-17-24-21.txt"];

hotpixel_filenames = ["dvs_vpr_2020-04-21-17-03-03_hot_pixels.txt";
                      "dvs_vpr_2020-04-22-17-24-21_hot_pixels.txt"];


filenames_cell = cell(length(filenames), 1);
for i = 1:length(filenames)
    filenames_cell{i} = [filenames(i), hotpixel_filenames(i)];
end



%% Run
%---- Loop through reading and saving data ----%
prev_interval_time_print = 0;

for i = 1:length(filenames) 
    prev_interval_time_print = 0;

    %---- Open the complete dataset ----%
    filename = filenames_cell{i}(1);
    fid_data = fopen(filename, 'r'); % open data set as read only
    
    % read the first line (discard - doesn't have useful information)
    tline = fgetl(fid_data); % first line of .txt file (doesn't have useful information)

    %---- Open file to save compressed data stream ----%
    if isfolder(save_dir+filename)
        continue_command = input("Save file already exists, if you continue it will be overwritten: Continue[1]/Exit[0]");
        if continue_command == '1'
            fid_compression = fopen(save_dir+filename, 'w'); % create and open a new file
        end
    else
        fid_compression = fopen(save_dir+filename, 'w'); % create and open a new file
    end

    % load the hot pixel data
    hotpixels = table2array(readtable(filenames_cell{i}(2)));
         
    % Arrays for spatial compression indexes
    y_compressed_indexes = linspace(0, rows, resolution(1) + 1);
    x_compressed_indexes = linspace(0, cols, resolution(2) + 1);
    
    iter = 1;
    while true
       
        tline = fgetl(fid_data); % read a line from the full dataset

        %---- If the line is valid -> process data ----%
        if ischar(tline)
            % split the text file line into its (t, x, y, pol) components
            data = split(tline);
            format long;
            t = str2double(data{1});
            x = str2double(data{2});
            y = str2double(data{3});
            pol = str2double(data{4});

            if iter == 1
                t_0 = t; 
                iter = iter + 1;
            end
               
            %---- Save if not a hot pixel ----%
            % check if the event occurs at a hot pixel
            pixel_coords = [x, y];
            is_hot_pixel = ismember(pixel_coords, hotpixels, 'rows');
            
            if ~(is_hot_pixel) % process if not a hot pixel 

                % find coords in the down sampled resolution
                x_ind = find(x_compressed_indexes <= x, 1, 'last') - 1; % to offset matlab indexing
                y_ind = find(y_compressed_indexes <= y, 1, 'last') - 1; % to offset matlab indexing
                
                % create string with new values and write to compression data file
                new_data_string = sprintf('%.15d, %d, %d, %d\n', t, x_ind, y_ind, pol);
                % fprintf(new_data_string);
                fprintf(fid_compression, new_data_string);
            end

            % print time every second to show that it is progressing
            % interval_time_print = (t - t_0) - prev_interval_time_print;
            % if interval_time_print > 1
            %     prev_interval_time_print = prev_interval_time_print + interval_time_print;
            %     interval_time_print = 0;
            %     fprintf("Time: \t %.2f\n", t-t_0);
            % end
        
        %---- If the line is invalid -> end of text file, break ----%
        else 
            break;
        end
    end

    fprintf("%s complete \n", filename); % print that the file is complete

    %---- Close the current text files ----%
    fclose(fid_data);
    fclose(fid_compression);
end

