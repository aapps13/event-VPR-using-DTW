%% Setup data hierarchy
clearvars 
linux = 1;

if linux
    data_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/';
    save_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/';
    % Get resolution from directory name
    data_dir_split = split(data_dir, '/');
    resolution_split = split(data_dir_split{end-1}, '-');
    resolution = str2double(resolution_split{2}(1:end-1));
else
    data_dir = 'F:\Honours\spatial_compression\resolution_[100-100]\';
    save_dir = 'F:\Honours\spatial_compression\resolution_[100-100]\';

    % Get resolution from directory name    
    data_dir_split = split(data_dir, '\');
    resolution_split = split(data_dir_split{end-1}, '-');
    resolution = str2double(resolution_split{2}(1:end-1));
end

filenames = ["dvs_vpr_2020-04-21-17-03-03.txt";
             "dvs_vpr_2020-04-22-17-24-21.txt"];

% Create cell array that matches the resolution dimensions
accumulation_cell = cell(resolution, resolution);


%---- Parameters ----%
file_choice = 1;


%---- Variables ----#
for i = 1:length(filenames) 
    iteration = 0;
    prev_interval_time_print = 0;
    
    fid = fopen(data_dir+filenames(i));
    file = erase(filenames(i), '.txt');

    while true
            tline = fgetl(fid); % read a line from the full dataset
          
            %---- If the line is valid -> process data ----%
            if ischar(tline)
                format long
                data_split = split(tline, ',');
                t = str2double(data_split{1});
                x = str2double(data_split{2});
                y = str2double(data_split{3});
                pol = str2double(data_split{4});
    
                if iteration == 0 
                    t_0 = t;
                    iteration = iteration + 1;
                end
                
                if pol == 1
                    pol_signed = 1;
                else % pol == 0 or pol == -1
                    pol_signed = -1; 
                end
           
                cell_index = size(accumulation_cell{x+1,y+1},2)+1;
    
                % First entry, add event polarity
                if cell_index == 1
                    accumulation_cell{x+1,y+1}(1:2,cell_index) = [pol_signed; t]; 
                
                % For each measurement, add the current polarity to the
                % previous accumulated value and store with the timestamp
                else
                    new_val = accumulation_cell{x+1,y+1}(1,cell_index-1) + pol_signed;
                    accumulation_cell{x+1,y+1}(1:2, cell_index) = [new_val; t]; 
                end
    
                %print time every second to show that it is progressing
                interval_time_print = (t - t_0) - prev_interval_time_print;
                if interval_time_print > 1
                    prev_interval_time_print = prev_interval_time_print + interval_time_print;
                    interval_time_print = 0;
                    fprintf("Time: \t %.2f\n", t-t_0);
                end

            else
                break
            end
    end
    
    % Close the test file
    fclose(fid);

    % Save the accumulation cell
    save_filename = save_dir + file + "_bias_integration.mat";
    save(save_filename, "accumulation_cell");
    fprintf("%s complete \n", file);

    clear accumulation_cell
end

