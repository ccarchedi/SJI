function varargout = GUI_Sp_CCP(varargin)
% GUI_SP_CCP MATLAB code for GUI_Sp_CCP.fig
%      GUI_SP_CCP, by itself, creates a new GUI_SP_CCP or raises the existing
%      singleton*.
%
%      H = GUI_SP_CCP returns the handle to a new GUI_SP_CCP or the handle to
%      the existing singleton*.
%
%      GUI_SP_CCP('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI_SP_CCP.M with the given input arguments.
%
%      GUI_SP_CCP('Property','Value',...) creates a new GUI_SP_CCP or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_Sp_CCP_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_Sp_CCP_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI_Sp_CCP

% Last Modified by GUIDE v2.5 29-Mar-2018 23:23:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_Sp_CCP_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_Sp_CCP_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI_Sp_CCP is made visible.
function GUI_Sp_CCP_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI_Sp_CCP (see VARARGIN)
handles.NVGs = load('USA_NVG_parameters.mat');
% Choose default command line output for GUI_Sp_CCP
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI_Sp_CCP wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_Sp_CCP_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function Lat_In_Callback(hObject, eventdata, handles)
% hObject    handle to Lat_In (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Lat_In as text
%        str2double(get(hObject,'String')) returns contents of Lat_In as a double
Lat = str2double(get(hObject,'String'));
if isnan(Lat)
    set(hObject,'String','');
    ed = errordlg('Input must be a number','Error');
elseif Lat > 50 || Lat < 27
    set(hObject,'String','');
    ed = errordlg({'Input latitude is outside of contiguous US!'...
        '  (Check you are in degrees North?)'},'Error');
end
if exist('ed','var')
    set(ed, 'WindowStyle','modal');
end


handles.Lat = Lat;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Lat_In_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Lat_In (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Long_In_Callback(hObject, eventdata, handles)
% hObject    handle to Long_In (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Long_In as text
%        str2double(get(hObject,'String')) returns contents of Long_In as a double
Long = str2double(get(hObject,'String'));
if isnan(Long)
    set(hObject,'String','');
    ed = errordlg('Input must be a number','Error');
elseif Long < 68 || Long > 127
    set(hObject,'String','');
    ed = errordlg({'Input longitude is outside of contiguous US!'...
        '  (Check you are in degrees West?)'},'Error');
end
if exist('ed','var')
    set(ed, 'WindowStyle','modal');
end


handles.Long = -Long;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Long_In_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Long_In (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in GoButton.
function GoButton_Callback(hObject, eventdata, handles)
% hObject    handle to GoButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

cla(handles.USMap); cla(handles.Map); colorbar(handles.Map,'off');
set(handles.USMap,'Visible','Off'); set(handles.Map,'Visible','Off');
% Get location data
if ~isfield(handles, 'Lat') || ~isfield(handles,'Long') || ...
        sum(isnan([handles.Lat, handles.Long]))
    ed = errordlg('You haven''t selected a location!','Error');
    set(ed, 'WindowStyle','modal');
    return
end
Lat = handles.Lat; Lon = handles.Long;

% Vals to retrieve
retrieve_vals = [handles.DepthYN.Value, handles.VelContrastYN.Value, ...
    handles.WidthYN.Value, handles.AmpYN.Value, handles.BreadthYN.Value];


if sum(retrieve_vals) == 0
    ed = errordlg('You haven''t selected any data to return!','Error');
    set(ed, 'WindowStyle','modal');
    return
end

% Account for if haven't pressed any buttons
if ~isfield(handles,'WhichNVG'); handles.WhichNVG = 'Strongest'; end
switch handles.WhichNVG
    case 'Strongest'; NVG = handles.NVGs.StrongestNVG;
    case 'LAB'; NVG = handles.NVGs.LAB;
end

if ~isfield(handles,'Where'); handles.Where = 'Within50'; end
dists = sqrt((NVG(:,1) - Lat).^2 + (NVG(:,2) - Lon).^2);
switch handles.Where
    case 'SpotOn'
        [dist_off, iloc] = min(dists);
        if dist_off > 0.5
            ed = errordlg('No NVG picks within 50 km!','Error');
            set(ed, 'WindowStyle','modal');
            return
        end
    case 'Within50'
        iloc = find(dists<0.5);
        if isempty(iloc)
            ed = errordlg('No NVG picks within 50 km!','Error');
            set(ed, 'WindowStyle','modal');
            return
        end
end
        


if handles.DepthYN.Value
    dep = num2str(round(median(NVG(iloc,3))));
    if length(iloc)>1
        stdstr = [' ± ' num2str(round(std(NVG(iloc,3))*1e1)/1e1)];
    else; stdstr = '';
    end
    handles.Depth.String = [dep stdstr ' km'];
end
if handles.AmpYN.Value
    amp = num2str(round(median(NVG(iloc,4))*1e3)/1e3);
    if length(iloc)>1
        stdstr = [' ± ' num2str(round(std(NVG(iloc,4))*1e4)/1e4)];
    else; stdstr = '';
    end
    handles.Amp.String = [amp stdstr];
end
if handles.BreadthYN.Value
    breadth = num2str(round(median(NVG(iloc,5))*1e1)/1e1);
    if length(iloc)>1
        stdstr = [' ± ' num2str(round(std(NVG(iloc,5))*1e2)/1e2)];
    else; stdstr = '';
    end
    handles.Breadth.String = [breadth stdstr ' km'];
end
if handles.VelContrastYN.Value
    dv = num2str(round(median(NVG(iloc,6))));
    if length(iloc)>1
        stdstr = [' ± ' num2str(round(std(NVG(iloc,6))*1e1)/1e1)];
    else; stdstr = '';
    end
    handles.VelContrast.String = [dv stdstr ' %'];
end
if handles.WidthYN.Value
    wid = num2str(round(median(NVG(iloc,7))));
    if length(iloc)>1
        stdstr = [' ± ' num2str(round(std(NVG(iloc,7))*1e1)/1e1)];
    else; stdstr = '';
    end
    handles.Width.String = [wid stdstr ' km' ];
end

% And plot the map
cmap=[83 94 173; 165 186 232; 193 226 247;213 227 148; ...
    233 229 48; 229 150 37; 200 30 33]./255;

hold(handles.Map,'on');
if ~isfield(handles,'whichplot'); handles.whichplot = 'None'; end
switch handles.whichplot
    case 'None'; return
    case 'NVG Depth'; wp = 3; cax = [60 110]; cmap = flipud(cmap);
    case 'Relative Velocity Contrast'; wp = 6; cax = [2 12];
    case 'Gradient Width'; wp = 7; cax = [10 40]; 
    case 'Sp Phase Amplitude'; wp = 4; cax = [0.02 0.06];
    case 'Sp Phase Breadth'; wp = 5; cax = [5 40]; 
end
        


if length(iloc)>1; pltsz = 15; else; pltsz = 100; end
scatter(handles.Map,NVG(iloc,2),NVG(iloc,1),pltsz,NVG(iloc,wp),'filled');
plot(handles.Map,Lon, Lat, 'kp');
if license('test','map_toolbox')
    States = shaperead('usastatelo', 'UseGeoCoords', true);
    for k = 1:length(States)
        plot(handles.Map, States(k).Lon, States(k).Lat, '-','color',0.6*[1 1 1]);
    end
end
% The boundaries of the Fenneman and Johnson physiogeographic provinces
% Fenneman, N., Johnson, D., 1946. Physical divisions of the United States: US Geologi-
% cal Survey map prepared in cooperation with the physiographic commission. US
% Geological Survey. Scale 1:7,000,000.
for k = 1:length(handles.NVGs.Provinces)
    plot(handles.Map,handles.NVGs.Provinces(k).Lon, ...
        handles.NVGs.Provinces(k).Lat, '-','color',0.4*[1 1 1]); 
end
lims = [min([NVG(iloc,2); Lon])-1.25 max([NVG(iloc,2); Lon])+1.25 ...
    min([NVG(iloc,1); Lat])-0.75 max([NVG(iloc,1); Lat])+0.75];
xlim(handles.Map,lims(1:2));
ylim(handles.Map,lims(3:4));
if license('test','map_toolbox')
    daspect(handles.Map,[111.16, 111.16*distance(Lat,0,Lat,1), 1]);
else
    daspect(handles.Map,[111.16, 111.16*0.819, 1]);
end
c=colorbar(handles.Map,'location','southoutside');
caxis(handles.Map,cax); colormap(cmap);
xlabel(c,handles.whichplot); set(handles.Map,'Visible','On');
box(handles.Map,'on');


hold(handles.USMap,'on');
if license('test','map_toolbox')
    for k = 1:length(States)
        plot(handles.USMap, States(k).Lon, States(k).Lat, '-','color',0.6*[1 1 1]);
    end
end
for k = 1:length(handles.NVGs.Provinces)
    plot(handles.USMap,handles.NVGs.Provinces(k).Lon, ...
        handles.NVGs.Provinces(k).Lat, '-','color',0.4*[1 1 1]); 
end
xlim(handles.USMap,[-127 -68]); ylim(handles.USMap,[27 50]);
plot(handles.USMap,lims([1 1 2 2 1]),lims([3 4 4 3 3]),'b-','linewidth',2);
box(handles.USMap,'on'); set(handles.USMap,'Visible','On');
daspect(handles.USMap,[111.16, 111.16*0.819, 1]);
set(handles.USMap,'xtick',[]); set(handles.USMap,'ytick',[]); 



% --- Executes on button press in LAB.
function LAB_Callback(hObject, eventdata, handles)
% hObject    handle to LAB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LAB
if get(hObject,'Value'); handles.WhichNVG = 'LAB'; end
guidata(hObject, handles);


% --- Executes on button press in Strongest.
function Strongest_Callback(hObject, eventdata, handles)
% hObject    handle to Strongest (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Strongest
if get(hObject,'Value'); handles.WhichNVG = 'Strongest'; end
guidata(hObject, handles);


% --- Executes on button press in Within50.
function Within50_Callback(hObject, eventdata, handles)
% hObject    handle to Within50 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Within50
if get(hObject,'Value'); handles.Where = 'Within50'; end
guidata(hObject, handles);

% --- Executes on button press in SpotOn.
function SpotOn_Callback(hObject, eventdata, handles)
% hObject    handle to SpotOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of SpotOn
if get(hObject,'Value'); handles.Where = 'SpotOn'; end
guidata(hObject, handles);


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1
contents = cellstr(get(hObject,'String'));
handles.whichplot = contents{get(hObject,'Value')};
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
