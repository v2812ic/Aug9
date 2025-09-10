%% Plot DCM / DCMdot / CoM / VRP (REF vs REAL) + (REF con OFFSET) sobre tiempo solapado
close all; clear; clc;

% === CONFIGURACIÓN ===
plot_offsets = true;          % <<--- Booleano: si true, también plotea referencias con offset, el deltat
com_offset_gain = 0.00125;    % Ganancia para CoM: com_ref - gain * offset
offset_file = fullfile('experiment_data','offsets.txt');

% Paths
addpath("plot/yaml_tools");
if exist("plot_foot.m","file")~=2 && exist("plot/plot_foot.m","file")==2
    addpath("plot");
end
has_plot_foot = exist('plot_foot','file')==2;

%% Cargar último YAML (referencias)
dd = dir("experiment_data/*.yaml");
assert(~isempty(dd), 'No YAML found in experiment_data/. Call SaveSolution() first.');

[~, i] = max([dd.datenum]); %#ok<DATNM>
yaml_path = fullfile(dd(i).folder, dd(i).name);
fprintf('loading %s\n', yaml_path);

S = ReadYaml(yaml_path);

% Parámetros temporales
initial_time = S.temporal_parameters.initial_time;
final_time   = S.temporal_parameters.final_time;
t_ds         = S.temporal_parameters.t_ds; %#ok<NASGU>
t_ss         = S.temporal_parameters.t_ss; %#ok<NASGU>
t_transfer   = S.temporal_parameters.t_transfer; %#ok<NASGU>

% Contactos (actual)
curr_rfoot_contact_pos = cell2mat(S.contact.curr_right_foot.pos);
curr_rfoot_contact_ori = cell2mat(S.contact.curr_right_foot.ori);
curr_lfoot_contact_pos = cell2mat(S.contact.curr_left_foot.pos);
curr_lfoot_contact_ori = cell2mat(S.contact.curr_left_foot.ori);

% Pasos planificados (centros)
rfoot_contact_pos = cell2mat(S.contact.right_foot.pos);
lfoot_contact_pos = cell2mat(S.contact.left_foot.pos);

% Referencias
t_ref       = cell2mat(S.reference.time); t_ref = t_ref(:);
dcm_pos_ref = cell2mat(S.reference.dcm_pos);
dcm_vel_ref = cell2mat(S.reference.dcm_vel);
com_pos_ref = cell2mat(S.reference.com_pos);
com_vel_ref = cell2mat(S.reference.com_vel);
vrp_ref     = cell2mat(S.reference.vrp);

%% Cargar trayectorias REALES si existen
has_real = false;
ddr = dir("experiment_data/com_real_*.mat");
if ~isempty(ddr)
    [~, ir] = max([ddr.datenum]); %#ok<DATNM>
    mat_path = fullfile(ddr(ir).folder, ddr(ir).name);
    fprintf('loading reals %s\n', mat_path);
    R = load(mat_path); % espera: t_act, com_pos_act, com_vel_act (y opcional xi_act, vrp_act, b_used)

    assert(isfield(R,'t_act') && isfield(R,'com_pos_act') && isfield(R,'com_vel_act'), ...
        'MAT must contain at least t_act, com_pos_act, com_vel_act');

    t_act       = R.t_act(:);
    com_pos_act = R.com_pos_act;
    com_vel_act = R.com_vel_act;

    % Parámetro LIPM b
    if isfield(R,'b_used')
        b_used = R.b_used;
    else
        g = 9.81;
        zc = mean(com_pos_act(:,3));
        b_used = sqrt(max(zc,1e-3)/g);
    end

    % DCM (xi) y VRP reales
    if isfield(R,'xi_act')
        xi_act = R.xi_act;
    else
        xi_act = com_pos_act + b_used*com_vel_act;
    end

    % xi_dot por gradient (robusto a no uniformidad)
    if numel(t_act) >= 2
        xidot_act = zeros(size(xi_act));
        for k = 1:3
            xidot_act(:,k) = gradient(xi_act(:,k), t_act);
        end
    else
        xidot_act = zeros(size(xi_act));
    end

    if isfield(R,'vrp_act')
        vrp_act = R.vrp_act;
    else
        vrp_act = xi_act - b_used * xidot_act;
    end

    has_real = true;
end

%% Cargar OFFSETS si se desea
has_offsets = false;
if plot_offsets
    if exist(offset_file,'file')==2
        M = readmatrix(offset_file);
        assert(size(M,2) >= 4, 'offsets.txt debe tener 4 columnas: [ox oy oz t].');
        t_off = M(:,4);
        off_xyz = M(:,1:3);

        % Limpieza de NaNs y tiempo único/monótono
        valid = all(isfinite([t_off off_xyz]),2);
        t_off = t_off(valid);
        off_xyz = off_xyz(valid,:);
        [t_off, ia] = unique(t_off,'stable');
        off_xyz = off_xyz(ia,:);

        if numel(t_off) < 2
            warning('offsets.txt no tiene suficientes muestras para interpolar. Se ignoran offsets.');
        else
            has_offsets = true;
        end
    else
        warning('No se encontró %s. Se ignoran offsets.', offset_file);
    end
end

%% Ventana temporal: SOLO SOLAPE
if has_real
    t0 = max(initial_time, min(t_act));
    t1 = min(final_time,   max(t_act));

    if t1 <= t0
        warning('No hay solape temporal entre referencias y reales (t0=%.3f, t1=%.3f).', t0, t1);
        % Aún así, dejamos referencias recortadas a [initial_time, final_time]
        t0 = initial_time; t1 = final_time;
    end

    mask_ref  = (t_ref >= t0) & (t_ref <= t1);
    mask_real = (t_act >= t0) & (t_act <= t1);

    % Recorte referencias
    t_ref_c        = t_ref(mask_ref);
    dcm_pos_ref_c  = dcm_pos_ref(mask_ref,:);
    dcm_vel_ref_c  = dcm_vel_ref(mask_ref,:);
    com_pos_ref_c  = com_pos_ref(mask_ref,:);
    com_vel_ref_c  = com_vel_ref(mask_ref,:);
    vrp_ref_c      = vrp_ref(mask_ref,:);

    % Recorte reales
    if any(mask_real)
        t_act_c    = t_act(mask_real);
        com_pos_c  = com_pos_act(mask_real,:);
        com_vel_c  = com_vel_act(mask_real,:);
        xi_act_c   = xi_act(mask_real,:);
        xidot_c    = xidot_act(mask_real,:);
        vrp_act_c  = vrp_act(mask_real,:);
    else
        % Sin solape real (por seguridad)
        t_act_c=[]; com_pos_c=[]; com_vel_c=[]; xi_act_c=[]; xidot_c=[]; vrp_act_c=[];
    end
else
    % Sin reales -> usar ventana completa [initial_time, final_time]
    t0 = initial_time;
    t1 = final_time;
    mask_ref = (t_ref >= t0) & (t_ref <= t1);

    t_ref_c       = t_ref(mask_ref);
    dcm_pos_ref_c = dcm_pos_ref(mask_ref,:);
    dcm_vel_ref_c = dcm_vel_ref(mask_ref,:);
    com_pos_ref_c = com_pos_ref(mask_ref,:);
    com_vel_ref_c = com_vel_ref(mask_ref,:);
    vrp_ref_c     = vrp_ref(mask_ref,:);
end

%% Interpolar y aplicar OFFSETS sobre la malla de referencia recortada
if has_offsets && ~isempty(t_ref_c)
    off_c = interp1(t_off, off_xyz, t_ref_c, 'linear', 'extrap'); % [N x 3]
    % Aplicaciones
    com_pos_ref_off_c = com_pos_ref_c - com_offset_gain * off_c; % posición con ganancia
    dcm_pos_ref_off_c = dcm_pos_ref_c - off_c;                    % DCM con offset tal cual
    vrp_ref_off_c     = vrp_ref_c     - off_c;                    % VRP con offset tal cual
    com_vel_ref_off_c = com_vel_ref_c - off_c;                    % *** NUEVO: vel CoM con offset (sin dt ni escalado)
else
    com_pos_ref_off_c = [];
    dcm_pos_ref_off_c = [];
    vrp_ref_off_c     = [];
    com_vel_ref_off_c = [];
end

%% XY: CoM / DCM / VRP (REF sólido) + (REAL dashed) + (REF con OFFSET punteado)
figure('Name','XY: CoM / DCM / VRP'); hold on; grid on; axis equal;

% CoM / DCM referencias
p1 = plot(com_pos_ref_c(:,1), com_pos_ref_c(:,2), 'Color',[0.85 0.33 0.10], 'LineWidth',2,  'DisplayName','CoM ref');
p2 = plot(dcm_pos_ref_c(:,1), dcm_pos_ref_c(:,2), 'Color',[0.93 0.69 0.13], 'LineWidth',2.5,'DisplayName','DCM ref');

% REF con offset (punteado)
if has_offsets
    p1o = plot(com_pos_ref_off_c(:,1), com_pos_ref_off_c(:,2), ':', 'Color',[0.85 0.33 0.10], 'LineWidth',1.8, 'DisplayName','CoM ref (offset)');
    p2o = plot(dcm_pos_ref_off_c(:,1), dcm_pos_ref_off_c(:,2), ':', 'Color',[0.93 0.69 0.13], 'LineWidth',2.0, 'DisplayName','DCM ref (offset)');
end

% REALES
if has_real && ~isempty(t_act_c)
    p1r = plot(com_pos_c(:,1), com_pos_c(:,2), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.5, 'DisplayName','CoM real');
    p2r = plot(xi_act_c(:,1),  xi_act_c(:,2),  '--', 'Color',[0.10 0.10 0.10], 'LineWidth',1.5, 'DisplayName','DCM real');
end

% Pies y pasos planificados (contexto)
if has_plot_foot
    plot_foot(gca, curr_lfoot_contact_pos, curr_lfoot_contact_ori, 'red');
    plot_foot(gca, curr_rfoot_contact_pos, curr_rfoot_contact_ori, 'blue');
else
    plot(curr_lfoot_contact_pos(1), curr_lfoot_contact_pos(2),'rs','MarkerFaceColor','r');
    plot(curr_rfoot_contact_pos(1), curr_rfoot_contact_pos(2),'bs','MarkerFaceColor','b');
end
if ~isempty(rfoot_contact_pos), plot(rfoot_contact_pos(:,1), rfoot_contact_pos(:,2), 'bo'); end
if ~isempty(lfoot_contact_pos), plot(lfoot_contact_pos(:,1), lfoot_contact_pos(:,2), 'ro'); end

% Leyenda dinámica
legend_handles = [p1 p2];
legend_texts   = {'CoM ref','DCM ref'};
if has_offsets
    legend_handles = [legend_handles p1o p2o];
    legend_texts   = [legend_texts {'CoM ref (offset)','DCM ref (offset)'}];
end
if exist('p1r','var') && exist('p2r','var')
    legend_handles = [legend_handles p1r p2r];
    legend_texts   = [legend_texts {'CoM real','DCM real'}];
end
legend(legend_handles, legend_texts, 'Location','best');

xlabel('x [m]'); ylabel('y [m]');
title(sprintf('XY (solape %0.2f–%0.2f s)', t0, t1));

%% Etiquetas
labs_pos   = {'x [m]','y [m]','z [m]'};
labs_vel   = {'x [m/s]','y [m/s]','z [m/s]'};

%% DCM vs tiempo (incluye ref con offset si existe)
figure('Name','DCM vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, dcm_pos_ref_c(:,k), 'LineWidth',1.6, 'DisplayName','DCM ref');
    if has_offsets
        plot(t_ref_c, dcm_pos_ref_off_c(:,k), ':', 'LineWidth',1.6, 'DisplayName','DCM ref (offset)');
    end
    if has_real && ~isempty(t_act_c)
        plot(t_act_c, xi_act_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2, 'DisplayName','DCM real');
    end
    xlim([t0 t1]);
    ylabel(labs_pos{k});
    if k==1, title('DCM (ref sólido, ref offset punteado, real dashed)'); legend('Location','best'); end
    if k==3, xlabel('time [s]'); end
end

%% DCMdot vs tiempo (sin offset explícito)
figure('Name','DCMdot vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, dcm_vel_ref_c(:,k), 'LineWidth',1.6, 'DisplayName','DCMdot ref');
    if has_real && ~isempty(t_act_c)
        plot(t_act_c, xidot_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2, 'DisplayName','DCMdot real');
    end
    xlim([t0 t1]);
    ylabel(labs_vel{k});
    if k==1, title('DCMdot (ref vs real)'); legend('Location','best'); end
    if k==3, xlabel('time [s]'); end
end

%% CoM vs tiempo (incluye ref con offset si existe)
figure('Name','CoM vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, com_pos_ref_c(:,k), 'LineWidth',1.6, 'DisplayName','CoM ref');
    if has_offsets
        plot(t_ref_c, com_pos_ref_off_c(:,k), ':', 'LineWidth',1.6, 'DisplayName','CoM ref (offset)');
    end
    if has_real && ~isempty(t_act_c)
        plot(t_act_c, com_pos_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2, 'DisplayName','CoM real');
    end
    xlim([t0 t1]);
    ylabel(labs_pos{k});
    if k==1, title('CoM (ref sólido, ref offset punteado, real dashed)'); legend('Location','best'); end
    if k==3, xlabel('time [s]'); end
end

%% *** NUEVO *** Velocidad de CoM vs tiempo (incluye ref con offset y real)
figure('Name','CoM velocity vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, com_vel_ref_c(:,k), 'LineWidth',1.6, 'DisplayName','CoM vel ref');
    if has_offsets
        plot(t_ref_c, com_vel_ref_off_c(:,k), ':', 'LineWidth',1.6, 'DisplayName','CoM vel ref (offset)');
    end
    if has_real && ~isempty(t_act_c)
        plot(t_act_c, com_vel_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2, 'DisplayName','CoM vel real');
    end
    xlim([t0 t1]);
    ylabel(labs_vel{k});
    if k==1, title('CoM velocity (ref sólido, ref offset punteado, real dashed)'); legend('Location','best'); end
    if k==3, xlabel('time [s]'); end
end

%% VRP vs tiempo (incluye ref con offset si existe)
figure('Name','VRP vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, vrp_ref_c(:,k), 'LineWidth',1.6, 'DisplayName','VRP ref');
    if has_offsets
        plot(t_ref_c, vrp_ref_off_c(:,k), ':', 'LineWidth',1.6, 'DisplayName','VRP ref (offset)');
    end
    if has_real && ~isempty(t_act_c)
        plot(t_act_c, vrp_act_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2, 'DisplayName','VRP real');
    end
    xlim([t0 t1]);
    ylabel(labs_pos{k});
    if k==1, title('VRP (ref sólido, ref offset punteado, real dashed)'); legend('Location','best'); end
    if k==3, xlabel('time [s]'); end
end
