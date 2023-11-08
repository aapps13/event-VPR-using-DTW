%% Setup data hierarchy
clearvars 

linux = 0;
if linux
    % data_dir = ...
    % save_dir = ...
else
    % % USB
    % data_dir = 'F:\Honours\spatial_compression\resolution_[100-100]\';
    % save_dir = 'F:\Honours\spatial_compression\resolution_[100-100]\';
    % % HARDRIVE
    data_dir = 'F:\Data\Output_Data\full_dataset\spatial_compression\resolution_[100-100]\';
    save_dir = 'F:\Data\Output_Data\full_dataset\spatial_compression\resolution_[100-100]\';

    % data_dir = 'F:\Data\Brisbane_Event_Data\';
    % save_dir = 'F:\Data\Brisbane_Event_Data\';
end

filenames = ["dvs_vpr_2020-04-21-17-03-03.txt";
             "dvs_vpr_2020-04-22-17-24-21.txt"];

%---- Variables ----#
for i = 1:length(filenames)
    % Setup variables for the loop
    events_per_second = [];
    iteration = 0;
    prev_interval_time = 0;
    event_counter = 0;
    array_index = 1;

    fid = fopen(data_dir+filenames(i));
    file = erase(filenames(i), '.txt');
    tline_first = fgetl(fid); % first line isn't useful
    fprintf('Processing %s \n', filenames(i))
    
    while true
            tline = fgetl(fid); % read a line from the full dataset
          
            %---- If the line is valid -> process data ----%
            if ischar(tline)
                % Read data
                data_split = split(tline);
                format long;
                t = str2double(data_split{1});
                x = str2double(data_split{2});
                y = str2double(data_split{3});
                pol = str2double(data_split{4});
                % Increment event counter
                event_counter = event_counter + 1;
    
                if iteration == 0 
                    t_0 = t;
                end
   
                % print time every second to show that it is progressing
                interval_time = (t - t_0) - prev_interval_time;
                
                if interval_time > 1    
                    % add data to array and reset variables
                    events_per_second(array_index) = event_counter;
                    event_counter = 0;
                    array_index = array_index + 1; 
                    prev_interval_time = prev_interval_time + interval_time;
                    interval_time = 0;
                    fprintf("Time: \t %.2f\n", t-t_0);
                end

                iteration = iteration + 1;
                
                % Just process the first 3 minutes -> should give enough data
                if (t-t_0 > 180)
                    break
                end

            else
                break
            end
    end
    
    % Close the test file
    fclose(fid);

    % Save the accumulation cell
    save_filename = save_dir + file + "_event_rate.mat";
    save(save_filename, "events_per_second");
    fprintf("%s complete \n", file);
end

