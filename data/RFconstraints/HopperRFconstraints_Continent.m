%%%%%%MATLAB code from Emily Hopper to make RF file for her code

% Convert CCP depths to times
clearvars; close all;% clc
%load('USA\Final Draft\Final Revision\GUI\USA_NVG_parameters.mat')
load('USA_NVG_parameters.mat')

CPlatlons = [25, 52, -125, -60];
grid_spacing = 0.25;
NVG = StrongestNVG; % LAB;

vpvs = 1.76;

lats = CPlatlons(1):grid_spacing:CPlatlons(2);
lons = CPlatlons(3):grid_spacing:CPlatlons(4);
[glon, glat] = meshgrid(lons, lats);

glon = glon(:);
glat = glat(:);

%basedir = 'C:/Users/Emily/OneDrive/Documents/WORK/MATLAB/Scattered_Waves/';
basedir = './';
%load([basedir 'Data\Velocity_Models\Mantle\Vs\SchmandtComboMoho.mat'])

BreadthInterp = scatteredInterpolant(StrongestNVG(:, 2), StrongestNVG(:, 1), StrongestNVG(:, 7));

SRz       = ncread('US.2016.nc', 'depth');
shen_lons = wrapTo180(ncread('US.2016.nc', 'longitude'));
shen_lats = ncread('US.2016.nc', 'latitude');
SRvsv     = ncread('US.2016.nc', 'vsv');

Moho_times = zeros(size(glat));
Moho_times_std = zeros(size(glat));
Moho_dv = zeros(size(glat));
Moho_dv_std = zeros(size(glat));
Moho_deps = Moho_dv; Moho_dep_stds = Moho_dv;
v_aboves = Moho_dv; v_belows = Moho_dv;

for k = 1:numel(glat)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [~, i_vlat] = min(abs(shen_lats - glat(k)));
    [~, i_vlon] = min(abs(shen_lons - glon(k)));

    vel_model = squeeze(SRvsv(i_vlon, i_vlat, :));

    dists = sqrt((NVG(:,1) - glat(k)).^2 + (NVG(:,2) - glon(k)).^2);
    iloc = find(dists<0.25);

    if isempty(iloc) || any(vel_model > 10)%flag in this model for no data

        travel_times(k) = nan;
        travel_times_std(k) = nan;
        amplitudes(k) = nan;
        amplitudes_std(k) = nan;
        breadth(k) = nan;

        continue

    end

    % convert to time
    dv = abs(diff(vel_model));
    
    %get top 2 and take the deeper on
    [~,l, ~, p] = findpeaks(dv);
    [p, pind]   = sort(p, 'descend');
    l           = l(pind);%sort locations of peak by their height

    discs = max(l(1:2));

    Moho_dep = discs*0.5 - 0.5;

    dep = median(NVG(iloc, 3));
    dep_p = dep + std(NVG(iloc, 3));
    dep_m = dep - std(NVG(iloc, 3));

    amplitudes(k)     = median(NVG(iloc, 4));
    amplitudes_std(k) = std(NVG(iloc,4));

    breadth(k) = BreadthInterp(glon(k), glat(k));
    
    i_dep_m = find(SRz < dep_m, 1, 'last');
    i_dep = find(SRz < dep, 1, 'last');
    travel_time_p = 0; travel_time_m = 0;
    for i_d = 1:length(SRz)
        travel_time_p = travel_time_p + 0.5/vel_model(i_d);
        if i_d == i_dep_m
            travel_time_m = travel_time_p;
        end
        if i_d == i_dep
            travel_times(k) = travel_time_p;
        end
    end
    travel_times_std(k) = mean([(travel_times(k) - travel_time_m), ...
        (travel_time_p - travel_times(k))]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Moho_dep_std = 1;

    i_dep_m = find(SRz < Moho_dep - Moho_dep_std, 1, 'last');
    i_dep = find(SRz < Moho_dep, 1, 'last');
    travel_time_p = 0; travel_time_m = 0;
    for i_d = 1:length(SRz)
        travel_time_p = travel_time_p + 0.5/vel_model(i_d);
        if i_d == i_dep_m
            travel_time_m = travel_time_p;
        end
        if i_d == i_dep
            Moho_times(k) = travel_time_p;
        end
    end
    Moho_times_std(k) = 0.6724;

    Moho_dv(k)     = vel_model(discs+1)/vel_model(discs) - 1;
    Moho_dv_std(k) = 0.0443;

end

clearvars -except glat glon travel_times travel_times_std amplitudes amplitudes_std Moho_times Moho_times_std Moho_dv Moho_dv_std breadth vpvs

%%
% Print to .csv file
%fid = fopen('D:\VM_shared\a_priori_constraints.csv', 'w'); % C:\Users\Emily\VM Shared\
fid = fopen('./a_priori_constraints_Continent.csv', 'w'); % C:\Users\Emily\VM Shared\
fprintf(fid, ...
    'lat,lon,ttMoho,ttMohostd,dvMoho,dvMohostd,breadthMoho,typeMoho,ttLAB,ttLABstd,ampLAB,ampLABstd,breadthLAB,typeLAB');
for k = 1:numel(glat)
    if isnan(travel_times(k)); continue; end
    fprintf(fid, '\n%g,%g,%g,%g,%g,%g,%g,%s,%g,%g,%g,%g,%g,%s', ...
       glat(k), glon(k), Moho_times(k), Moho_times_std(k), Moho_dv(k), ...
       Moho_dv_std(k), 1, 'Ps', travel_times(k)/vpvs, travel_times_std(k)/vpvs, ...
       amplitudes(k), amplitudes_std(k), breadth(k), 'Sp');

end

fclose(fid);
