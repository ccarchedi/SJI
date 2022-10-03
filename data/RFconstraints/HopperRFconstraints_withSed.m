%%%%%%MATLAB code from Emily Hopper to make RF file for her code

% Convert CCP depths to times
clearvars; close all; clc
%load('USA\Final Draft\Final Revision\GUI\USA_NVG_parameters.mat')
load('USA_NVG_parameters.mat')

CPlatlons = [30, 45, -119, -101];
grid_spacing = 0.25;
NVG = StrongestNVG; % LAB;


lats = CPlatlons(1):grid_spacing:CPlatlons(2);
lons = CPlatlons(3):grid_spacing:CPlatlons(4);
[glon, glat] = meshgrid(lons, lats);
%basedir = 'C:/Users/Emily/OneDrive/Documents/WORK/MATLAB/Scattered_Waves/';
basedir = './';
%load([basedir 'Data\Velocity_Models\Mantle\Vs\SchmandtComboMoho.mat'])

%%
% % inds = find((NVG > CPlatlons(1)) & (NVG < CPlatlons(2)) & ...
% %             (NVG > CPlatlons(3)) & (NVG < CPlatlons(4)));
% %
% cmap=[83 94 173; 165 186 232; 193 226 247;213 227 148; ...
%     233 229 48; 229 150 37; 200 30 33]./255;
% 
% deps_all = zeros(size(glat));
% deps_p = deps_all; deps_m = deps_all;
% 
% 
% travel_times = zeros(size(glat));
% travel_times_std = zeros(size(glat));
% amplitudes = zeros(size(glat));
% amplitudes_std = zeros(size(glat));
% for k = 1:numel(glat)
%     dists = sqrt((NVG(:,1) - glat(k)).^2 + (NVG(:,2) - glon(k)).^2);
%     iloc = find(dists<0.25);
%     if isempty(iloc)
%         travel_times(k) = nan;
%         travel_times_std(k) = nan;
%         amplitudes(k) = nan;
%         amplitudes_std(k) = nan;
%         continue
%     end
%     dep = median(NVG(iloc, 3));
%     dep_p = dep + std(NVG(iloc, 3));
%     dep_m = dep - std(NVG(iloc, 3));
% 
%     amplitudes(k) = median(NVG(iloc, 4));
%     amplitudes_std(k) = std(NVG(iloc,4));
% 
%     % convert to time
%     dep_spacing = 0.5;
%     deps = 0:dep_spacing:dep_p;
%     [~, i_vlat] = min(abs(Vs_Model.Latitude - glat(k)));
%     [~, i_vlon] = min(abs(Vs_Model.Longitude - glon(k)));
%     vels = interp1(Vs_Model.Depth, ...
%         reshape(Vs_Model.Vp(i_vlat, i_vlon, :), size(Vs_Model.Depth)), deps);
%     i_dep_m = find(deps < dep_m, 1, 'last');
%     i_dep = find(deps < dep, 1, 'last');
%     travel_time_p = 0; travel_time_m = 0;
%     for i_d = 1:length(deps)
%         travel_time_p = travel_time_p + dep_spacing/vels(i_d);
%         if i_d == i_dep_m
%             travel_time_m = travel_time_p;
%         end
%         if i_d == i_dep
%             travel_times(k) = travel_time_p;
%         end
%     end
%     travel_times_std(k) = mean([(travel_times(k) - travel_time_m), ...
%         (travel_time_p - travel_times(k))]);
% end

%%

% Get Moho values from Shen et al., 2012 WUS model

% http://ciei.colorado.edu/Models/US_4/Model_Description.pdf
% These are saved as individual text files, named '(lon E)_(lat N).mod.1'
% where lon E = 235.5:0.25:260; lat N = 28.5:0.25:49
% From their description:
% The first line is formatted:
% lon lat Crustal_thickness (km) Error_crustal_thickness (km)
% Then each line thereafter is: depth (km), Vsv (km/s), Error_Vsv (km/s).
% At layer boundaries (base of the sediments, base of the crust) there is
% a repeated knot so the model can take a discrete jump.

shen_lats = 28.5:0.25:49;
shen_lons = 235.5:0.25:260;

sed_times     = zeros(size(glat));
sed_times_std = zeros(size(glat));
sed_dv        = zeros(size(glat));
sed_dv_std    = zeros(size(glat));

Moho_times = zeros(size(glat));
Moho_times_std = zeros(size(glat));
Moho_dv = zeros(size(glat));
Moho_dv_std = zeros(size(glat));
Moho_deps = Moho_dv; Moho_dep_stds = Moho_dv;
v_aboves = Moho_dv; v_belows = Moho_dv;

for k = 1:numel(glat)
    disp(k)

    [~, i_vlat] = min(abs(shen_lats - glat(k)));
    [~, i_vlon] = min(abs((shen_lons - 360) - glon(k)));

    fname = [num2str(shen_lons(i_vlon)), '_', ...
        num2str(shen_lats(i_vlat)), '.mod.1'];
    %fid = fopen([basedir '/Data/Velocity_Models/WUSA/' fname]); jsb
    fid = fopen([basedir './WUSA/' fname]);
    fid1 = fid;
    while fid < 0
        i_vlat = i_vlat + 1;
        i_vlon = i_vlon + 1;
        fname = [num2str(shen_lons(i_vlon)), '_', ...
            num2str(shen_lats(i_vlat)), '.mod.1'];
        %fid = fopen([basedir '/Data/Velocity_Models/WUSA/' fname]); jsb
        fid = fopen([basedir './WUSA/' fname]); 
    end
    %     if fid1 < 0
    %         fprintf('\nOriginal lat/lon: %.2f, %.2f;\t\t difference: %.2f, %.2f', ...
    %             glat(k), glon(k), shen_lats(i_vlat)-glat(k), (shen_lons(i_vlon)-360)-glon(k));
    %     end

    Moho_data = strsplit(fgetl(fid));
    Moho_dep = str2double(Moho_data{3});
    Moho_dep_std = str2double(Moho_data{4});

    vel_model = fscanf(fid, '%g %g %g', [3 Inf]);
    
    
    % convert to time 
    discs = find(diff(vel_model(1,:)) == 0);
    vel_model(1, discs+1) = vel_model(1, discs+1) + 0.01;
    
    %%%%JSB get the sediment time and dv before doing Moho stuff
    
    dep_spacing = 0.01;
    deps = 0:dep_spacing:vel_model(1, discs(1)) + 0.25;
    vels = interp1(vel_model(1,:), vel_model(2,:), deps);
    i_dep_m = find(deps < vel_model(1, discs(1)) - 0.25, 1, 'last');
    i_dep = find(deps < vel_model(1, discs(1)), 1, 'last');
    travel_time_p = 0; travel_time_m = 0;
    for i_d = 1:length(deps)
        travel_time_p = travel_time_p + dep_spacing/vels(i_d);
        if i_d == i_dep_m
            travel_time_m = travel_time_p;
        end
        if i_d == i_dep
            sed_times(k) = travel_time_p;
        end
    end
    sed_times_std(k) = mean([(sed_times(k) - travel_time_m), ...
        (travel_time_p - sed_times(k))]);

    v_above     = vel_model(2,discs(1));
    v_above_std = vel_model(3, discs(1));
    v_below     = vel_model(2, discs(1)+1);
    v_below_std = vel_model(3, discs(1)+1);

    sed_dv(k) = v_below/v_above - 1;
    
    x = v_below / v_above ...
        * sqrt(...
            (v_below_std / v_below) ^ 2 ...
            + (v_above_std / v_above) ^ 2 ...
            - (0.0079 / (v_below * v_above)) ... # 0.0079 from cov(v_belows, v_aboves)
        );

    if isreal(x)
        
        sed_dv_std(k) = x;
        
    else
       
        sed_dv_std(k) = -1;
        
    end
    
    %%%%
    
    vel_model(1, discs+1) = vel_model(1, discs+1) + 0.01;
    dep_spacing = 0.5;
    deps = 0:dep_spacing:Moho_dep + Moho_dep_std;
    vels = interp1(vel_model(1,:), vel_model(2,:), deps);
    i_dep_m = find(deps < Moho_dep - Moho_dep_std, 1, 'last');
    i_dep = find(deps < Moho_dep, 1, 'last');
    travel_time_p = 0; travel_time_m = 0;
    for i_d = 1:length(deps)
        travel_time_p = travel_time_p + dep_spacing/vels(i_d);
        if i_d == i_dep_m
            travel_time_m = travel_time_p;
        end
        if i_d == i_dep
            Moho_times(k) = travel_time_p;
        end
    end
    Moho_times_std(k) = mean([(Moho_times(k) - travel_time_m), ...
        (travel_time_p - Moho_times(k))]);
    Moho_dep_stds(k) = Moho_dep_std;
    Moho_deps(k) = Moho_dep;


    ind = find(vel_model(1,:) == Moho_dep, 1);
    v_above = vel_model(2,ind);
    v_above_std = vel_model(3, ind);
    v_below = vel_model(2, ind+1);
    v_below_std = vel_model(3, ind+1);
    v_belows(k) = v_below;
    v_aboves(k) = v_above;
    %median(vel_model(3, 1:end -1) ./ vel_model(3, 2:end));

    Moho_dv(k) = v_below/v_above - 1; %fractional change, NOT total change
    
    % std for the division of two real variables A, B
    %   f = A/B
    % sigma_f = |f| * sqrt((sigma_A/A)^2) + (sigma_B/B)^2 - 2(sigma_AB/A/B))
    % from https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    % covariance, sigma_AB, actually expected to be pretty high (i.e. the
    % adjacent velocity values will be highly correlated), but unknown here
    % Perfectly correlated would be sigma_AB = sigma_A * sigma_B, which
    % is on average about 0.016.  Here, I've taken the covmat(2,1) value
    % from calculating the covariance of v_below and v_above across the
    % model space.
    
    %if ((v_below_std / v_below) ^ 2 + (v_above_std / v_above) ^ 2) < 2*(0.0079 / (v_below * v_above))
        
    %    keyboard
        
    %end
    
    %%%%%jsb investigation
    x = v_below / v_above ...
        * sqrt(...
            (v_below_std / v_below) ^ 2 ...
            + (v_above_std / v_above) ^ 2 ...
            - (0.0079 / (v_below * v_above)) ... # 0.0079 from cov(v_belows, v_aboves)
        );

    if isreal(x)
        
        Moho_dv_std(k) = x;
        
    else
       
        Moho_dv_std(k) = -1;
        
    end
    
    x = v_below / v_above ...
        * sqrt(...
            (v_below_std / v_below) ^ 2 ...
            + (v_above_std / v_above) ^ 2 ...
            - 2*(0.0079 / (v_below * v_above)) ... # 0.0079 from cov(v_belows, v_aboves)
        );

    if isreal(x)
        
        Moho_dv_std2(k) = x;
        
    else
       
        Moho_dv_std2(k) = -1;
        
    end
    
    fclose(fid);

end

covmat = cov(v_belows, v_aboves);
disp(covmat(2,1))

%%%%%jsb add to preexisting table
ttSed        = sed_times(:);
dvSed        = sed_dv(:);
dvSedstd     = sed_dv_std(:);
ttSedstd     = sed_times_std(:);
breadthSed   = 0.001*ones(size(sed_times_std(:)));
typeSed      = cell(size(ttSed));
[typeSed{:}] = deal('Ps');

sedTable = table(ttSed, ttSedstd, dvSed, dvSedstd, breadthSed, typeSed);

load('aprior_breadth.mat')

kill = zeros(size(ttSed));

for k = 1:length(kill)
   
    if isempty(find( (aprioriconstraintsbreadth.lat == glat(k)) & (aprioriconstraintsbreadth.lon == glon(k)))) %#ok<EFIND>
       
        kill(k) = 1;
        
    end
    
end

sedTable(logical(kill), :) = [];

writetable([ aprioriconstraintsbreadth sedTable ], 'a_priori_constraints_sed.csv', 'WriteVariableNames', true)