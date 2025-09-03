%% Plot DCM / DCMdot / CoM / VRP (REF vs REAL) over overlapping time only
close all; clear; clc;

% Paths
addpath("plot/yaml_tools");
if exist("plot_foot.m","file")~=2 && exist("plot/plot_foot.m","file")==2
    addpath("plot");
end

%% Load latest YAML (references)
dd = dir("experiment_data/*.yaml");
assert(~isempty(dd), 'No YAML found in experiment_data/. Call SaveSolution() first.');

[~, i] = max([dd.datenum]);
yaml_path = fullfile(dd(i).folder, dd(i).name);
fprintf('loading %s\n', yaml_path);

S = ReadYaml(yaml_path);

% Temporal params
initial_time = S.temporal_parameters.initial_time;
final_time   = S.temporal_parameters.final_time;
t_ds         = S.temporal_parameters.t_ds;
t_ss         = S.temporal_parameters.t_ss;
t_transfer   = S.temporal_parameters.t_transfer;

% Contacts (current)
curr_rfoot_contact_pos = cell2mat(S.contact.curr_right_foot.pos);
curr_rfoot_contact_ori = cell2mat(S.contact.curr_right_foot.ori);
curr_lfoot_contact_pos = cell2mat(S.contact.curr_left_foot.pos);
curr_lfoot_contact_ori = cell2mat(S.contact.curr_left_foot.ori);

% Planned footsteps (centers)
rfoot_contact_pos = cell2mat(S.contact.right_foot.pos);
lfoot_contact_pos = cell2mat(S.contact.left_foot.pos);

% References
t_ref       = cell2mat(S.reference.time); t_ref = t_ref(:);
dcm_pos_ref = cell2mat(S.reference.dcm_pos);
dcm_vel_ref = cell2mat(S.reference.dcm_vel);
com_pos_ref = cell2mat(S.reference.com_pos);
com_vel_ref = cell2mat(S.reference.com_vel);
vrp_ref     = cell2mat(S.reference.vrp); % ensure you DID NOT add +100 in C++

%% Load REAL trajectories if available
has_real = false;
ddr = dir("experiment_data/com_real_*.mat");
if ~isempty(ddr)
    [~, ir] = max([ddr.datenum]);
    mat_path = fullfile(ddr(ir).folder, ddr(ir).name);
    fprintf('loading reals %s\n', mat_path);
    R = load(mat_path); % expects: t_act, com_pos_act, com_vel_act (and optionally xi_act, vrp_act, b_used)

    assert(isfield(R,'t_act') && isfield(R,'com_pos_act') && isfield(R,'com_vel_act'), ...
        'MAT must contain at least t_act, com_pos_act, com_vel_act');

    t_act       = R.t_act(:);
    com_pos_act = R.com_pos_act;
    com_vel_act = R.com_vel_act;

    % LIPM parameter b
    if isfield(R,'b_used')
        b_used = R.b_used;
    else
        g = 9.81;
        zc = mean(com_pos_act(:,3));
        b_used = sqrt(max(zc,1e-3)/g);
    end

    % Real DCM (xi) and VRP
    if isfield(R,'xi_act')
        xi_act = R.xi_act;
    else
        xi_act = com_pos_act + b_used*com_vel_act;
    end

    % xi_dot via gradient per component (robust against spacing issues)
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

%% Optional: ICP from /tmp (kept as-is)
has_icp = false;
dd_kf = dir("/tmp/draco_state_estimator_kf_data*.mat");
if ~isempty(dd_kf)
    [~, ikf] = max([dd_kf.datenum]);
    kf_path = fullfile(dd_kf(ikf).folder, dd_kf(ikf).name);
    fprintf('loading %s\n', kf_path);
    K = load(kf_path, 'icp_est');
    if isfield(K,'icp_est')
        icp_est = K.icp_est;
        has_icp = true;
    end
end

%% Time window: OVERLAP ONLY
if has_real
    t0 = max(initial_time, min(t_act));
    t1 = min(final_time,   max(t_act));

    mask_ref  = (t_ref >= t0) & (t_ref <= t1);
    mask_real = (t_act >= t0) & (t_act <= t1);

    % Crop references
    t_ref_c        = t_ref(mask_ref);
    dcm_pos_ref_c  = dcm_pos_ref(mask_ref,:);
    dcm_vel_ref_c  = dcm_vel_ref(mask_ref,:);
    com_pos_ref_c  = com_pos_ref(mask_ref,:);
    com_vel_ref_c  = com_vel_ref(mask_ref,:);
    vrp_ref_c      = vrp_ref(mask_ref,:);

    % Crop reals
    t_act_c    = t_act(mask_real);
    com_pos_c  = com_pos_act(mask_real,:);
    com_vel_c  = com_vel_act(mask_real,:);
    xi_act_c   = xi_act(mask_real,:);
    xidot_c    = xidot_act(mask_real,:);
    vrp_act_c  = vrp_act(mask_real,:);

else
    % No reals -> keep full references (still respect initial_time if you want)
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

%% XY: CoM / DCM / VRP (REF solid) + (REAL dashed) over overlap only
figure('Name','XY: CoM / DCM / VRP (overlap only)'); hold on; grid on; axis equal;
p1 = plot(com_pos_ref_c(:,1), com_pos_ref_c(:,2), 'Color',[0.85 0.33 0.10], 'LineWidth',2);      % CoM ref
p2 = plot(dcm_pos_ref_c(:,1), dcm_pos_ref_c(:,2), 'Color',[0.93 0.69 0.13], 'LineWidth',2.5);    % DCM ref
p3 = plot(vrp_ref_c(:,1),     vrp_ref_c(:,2),     'Color',[0.00 0.45 0.74], 'LineWidth',2);      % VRP ref
if has_real
    p1r = plot(com_pos_c(:,1), com_pos_c(:,2), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.5);
    p2r = plot(xi_act_c(:,1),  xi_act_c(:,2),  '--', 'Color',[0.10 0.10 0.10], 'LineWidth',1.5);
    p3r = plot(vrp_act_c(:,1), vrp_act_c(:,2), '--', 'Color',[0.00 0.45 0.74], 'LineWidth',1.5);
end

% Feet and planned steps (context)
if exist('plot_foot','file')==2
    plot_foot(gca, curr_lfoot_contact_pos, curr_lfoot_contact_ori, 'red');
    plot_foot(gca, curr_rfoot_contact_pos, curr_rfoot_contact_ori, 'blue');
else
    plot(curr_lfoot_contact_pos(1), curr_lfoot_contact_pos(2),'rs','MarkerFaceColor','r');
    plot(curr_rfoot_contact_pos(1), curr_rfoot_contact_pos(2),'bs','MarkerFaceColor','b');
end
if ~isempty(rfoot_contact_pos), plot(rfoot_contact_pos(:,1), rfoot_contact_pos(:,2), 'bo'); end
if ~isempty(lfoot_contact_pos), plot(lfoot_contact_pos(:,1), lfoot_contact_pos(:,2), 'ro'); end

if has_real
    legend([p1 p2 p3 p1r p2r p3r], ...
        {'CoM ref','DCM ref','VRP ref','CoM real','DCM real','VRP real'}, 'Location','best');
else
    legend([p1 p2 p3], {'CoM ref','DCM ref','VRP ref'}, 'Location','best');
end
xlabel('x [m]'); ylabel('y [m]');
title(sprintf('XY trajectories (overlap %0.2f–%0.2f s)', t0, t1));

%% Axis labels
axis_names = {'x','y','z'};
labs_pos   = {'x [m]','y [m]','z [m]'};
labs_vel   = {'x [m/s]','y [m/s]','z [m/s]'};

%% DCM vs time (overlap)
figure('Name','DCM vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, dcm_pos_ref_c(:,k), 'LineWidth',1.6);
    if has_real
        plot(t_act_c, xi_act_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2);
    end
    xlim([t0 t1]);
    ylabel(labs_pos{k});
    if k==1, title('DCM components (ref solid vs. real dashed)'); end
    if k==3, xlabel('time [s]'); end
    if has_real && k==1
        legend('DCM ref','DCM real','Location','best');
    end
end

%% DCMdot vs time (overlap)
figure('Name','DCMdot vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, dcm_vel_ref_c(:,k), 'LineWidth',1.6);
    if has_real
        plot(t_act_c, xidot_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2);
    end
    xlim([t0 t1]);
    ylabel(labs_vel{k});
    if k==1, title('DCMdot components (ref solid vs. real dashed)'); end
    if k==3, xlabel('time [s]'); end
    if has_real && k==1
        legend('DCMdot ref','DCMdot real','Location','best');
    end
end

%% CoM vs time (overlap)
figure('Name','CoM vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, com_pos_ref_c(:,k), 'LineWidth',1.6);
    if has_real
        plot(t_act_c, com_pos_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2);
    end
    xlim([t0 t1]);
    ylabel(labs_pos{k});
    if k==1, title('CoM components (ref solid vs. real dashed)'); end
    if k==3, xlabel('time [s]'); end
    if has_real && k==1
        legend('CoM ref','CoM real','Location','best');
    end
end

%% VRP vs time (overlap)
figure('Name','VRP vs time (overlap only)');
for k=1:3
    subplot(3,1,k); hold on; grid on;
    plot(t_ref_c, vrp_ref_c(:,k), 'LineWidth',1.6);
    if has_real
        plot(t_act_c, vrp_act_c(:,k), '--', 'Color',[0.85 0.33 0.10], 'LineWidth',1.2);
    end
    xlim([t0 t1]);
    ylabel(labs_pos{k});
    if k==1, title('VRP components (ref solid vs. real dashed)'); end
    if k==3, xlabel('time [s]'); end
    if has_real && k==1
        legend('VRP ref','VRP real','Location','best');
    end
end
