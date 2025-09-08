#include "controller/g1_controller/g1_state_machines/move_foot_backward.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/g1_controller/g1_tci_container.hpp"
#include "controller/g1_controller/g1_task/g1_com_xy_task.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "controller/whole_body_controller/managers/end_effector_trajectory_manager.hpp"
#include "controller/whole_body_controller/managers/floating_base_trajectory_manager.hpp"
#include "util/interpolation.hpp"
#include "controller/whole_body_controller/managers/max_normal_force_trajectory_manager.hpp"
#include "controller/whole_body_controller/managers/reaction_force_trajectory_manager.hpp"
#include "controller/whole_body_controller/managers/task_hierarchy_manager.hpp"
#include <cmath>

MoveFootBackward::MoveFootBackward(const StateId state_id,
                                           PinocchioRobotSystem *robot,
                                           G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch) {
  util::PrettyConstructor(2, "MovingFootBackward");
  nominal_lfoot_iso_.setIdentity();
  nominal_rfoot_iso_.setIdentity();
  sp_ = G1StateProvider::GetStateProvider();
}

void MoveFootBackward::FirstVisit() {
  std::cout << "g1_states: Moving Foot Backward" << std::endl;
  state_machine_start_time_ = sp_->current_time_;
  
  end_time_ = state_machine_start_time_ + 2*swing_semitime_;

  swing_is_left_ = (lift_foot_ == 0);
  if (swing_is_left_){
    sp_->b_swing_leg_ = end_effector::LFoot;
  } else {
    sp_->b_swing_leg_ = end_effector::RFoot;
  }

  // Ya esta hecho de la anterior tarea, no lo tocamos
  //auto com_xy_task = ctrl_arch_->tci_container_->task_map_["com_xy_task"];
  //com_xy_ini = com_xy_task->DesiredPos();
  
  {/*
  if (swing_is_left_) {
    Eigen::Isometry3d stance_foot_iso = robot_->GetLinkIsometry(g1_link::r_foot_contact);
    ctrl_arch_->lf_pos_hm_->InitializeRampToMin(sway_time_);
    ctrl_arch_->lf_ori_hm_->InitializeRampToMin(sway_time_);
    ctrl_arch_->lf_max_normal_froce_tm_->InitializeRampToMin(sway_time_);
    ctrl_arch_->rf_pos_hm_->InitializeRampToMax(sway_time_);
    ctrl_arch_->rf_ori_hm_->InitializeRampToMax(sway_time_);
    ctrl_arch_->rf_max_normal_froce_tm_->InitializeRampToMax(sway_time_);
    sp_->b_lf_contact_ = false; // ojito 

  } else {
    Eigen::Isometry3d stance_foot_iso = robot_->GetLinkIsometry(g1_link::l_foot_contact);
    com_xy_target = stance_foot_iso.translation().head<2>();
    // Incluir rampa de fuerza
    ctrl_arch_->rf_pos_hm_->InitializeRampToMin(sway_time_);
    ctrl_arch_->rf_ori_hm_->InitializeRampToMin(sway_time_);
    ctrl_arch_->rf_max_normal_froce_tm_->InitializeRampToMin(sway_time_);
    ctrl_arch_->lf_pos_hm_->InitializeRampToMax(sway_time_);
    ctrl_arch_->lf_ori_hm_->InitializeRampToMax(sway_time_);
    ctrl_arch_->lf_max_normal_froce_tm_->InitializeRampToMax(sway_time_);
    sp_->b_rf_contact_ = false; // ojito 

  }
    */} // Todo esto ya está hecho de la otra tarea
  //com_xy_target = com_xy_ini;
  //com_xy_target += com_offset_;

  if (swing_is_left_) {
    Eigen::Isometry3d start_iso = robot_->GetLinkIsometry(g1_link::l_foot_contact);
    Eigen::Isometry3d final_iso = start_iso;
    final_iso.translation()= final_iso.translation() - foot_init_pos_swing_offset_ + foot_end_pos_swing_offset_;
 
    const double apex_z = start_iso.translation().z();

    if (not first_time_) swing_semitime_ *= 2;

    ctrl_arch_->lf_SE3_tm_->InitializeSwingTrajectory(start_iso, final_iso, apex_z, swing_semitime_);
    nominal_rfoot_iso_ = robot_->GetLinkIsometry(g1_link::r_foot_contact);

  } else {
    Eigen::Isometry3d start_iso = robot_->GetLinkIsometry(g1_link::r_foot_contact);
    Eigen::Isometry3d final_iso = start_iso;
    final_iso.translation() = final_iso.translation() - foot_init_pos_swing_offset_ + foot_end_pos_swing_offset_;
 
    const double apex_z = start_iso.translation().z();

    if (not first_time_) swing_semitime_ *= 2;

    ctrl_arch_->lf_SE3_tm_->InitializeSwingTrajectory(start_iso, final_iso, apex_z, swing_semitime_);

    nominal_rfoot_iso_ = robot_->GetLinkIsometry(g1_link::l_foot_contact);

  }
}

void MoveFootBackward::OneStep() {
  state_machine_time_ = sp_->current_time_ - state_machine_start_time_;
  swing_ref_ = state_machine_time_;
  swing_ref_ = std::clamp(swing_ref_, 0.0, end_time_);

{/*
  // 1) Interpolar CoM
  double s = util::SmoothPos(0, 1, sway_time_, state_machine_time_);
  s = std::clamp(s, 0.0, 1.0);
  Eigen::Vector2d com_xy_des = (1.0 - s) * com_xy_ini + s * com_xy_target;
  ctrl_arch_->tci_container_->task_map_["com_xy_task"]
      ->UpdateDesired(com_xy_des, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
      */}// No hace falta porque el centro de masas se quiere que esté quietecito, sin actualizar, en esta tarea

  // 2) Actualizar pies si cal - cal
  if (swing_is_left_) {
    ctrl_arch_->lf_SE3_tm_->UpdateDesired(swing_ref_);
    ctrl_arch_->rf_SE3_tm_->UseNominal(nominal_rfoot_iso_);
  } else {
    ctrl_arch_->rf_SE3_tm_->UpdateDesired(swing_ref_);
    ctrl_arch_->lf_SE3_tm_->UseNominal(nominal_lfoot_iso_);
  }

  // 3) Actualizar fuerzas de contanto
  {/*
  if (swing_is_left_){
    ctrl_arch_->lf_max_normal_froce_tm_->UpdateRampToMin(state_machine_time_);
    ctrl_arch_->lf_pos_hm_->UpdateRampToMin(state_machine_time_);
    ctrl_arch_->lf_ori_hm_->UpdateRampToMin(state_machine_time_);
    ctrl_arch_->rf_max_normal_froce_tm_->UpdateRampToMax(state_machine_time_);
    ctrl_arch_->rf_pos_hm_->UpdateRampToMax(state_machine_time_);
    ctrl_arch_->rf_ori_hm_->UpdateRampToMax(state_machine_time_);
  }
  else {
    ctrl_arch_->rf_max_normal_froce_tm_->UpdateRampToMin(state_machine_time_);
    ctrl_arch_->rf_pos_hm_->UpdateRampToMin(state_machine_time_);
    ctrl_arch_->rf_ori_hm_->UpdateRampToMin(state_machine_time_);
    ctrl_arch_->lf_max_normal_froce_tm_->UpdateRampToMax(state_machine_time_);
    ctrl_arch_->lf_pos_hm_->UpdateRampToMax(state_machine_time_);
    ctrl_arch_->lf_ori_hm_->UpdateRampToMax(state_machine_time_);
  }
    */} // No hacen falta aqui
}

bool MoveFootBackward::EndOfState() { 
  return (sp_->current_time_ >= end_time_ + wait_time_);
}

void MoveFootBackward::LastVisit() {
  // no haga nada y quito el flag de first_time
  first_time_ = false;
  state_machine_time_ = 0.;
}

StateId MoveFootBackward::GetNextState() {
  if (swing_foot_)
    return g1_states::MoveFootForward;
}

void MoveFootBackward::SetParameters(const YAML::Node &node) {
  try {
    util::ReadParameter(node["state_machine"]["swaying_foot"], "swing_semitime", swing_semitime_);
    util::ReadParameter(node["state_machine"]["swaying_foot"], "wait_time", wait_time_);
    util::ReadParameter(node["state_machine"]["swaying_foot"], "foot_init_pos_swing_offset", foot_init_pos_swing_offset_);
    util::ReadParameter(node["state_machine"]["swaying_foot"], "foot_end_pos_swing_offset", foot_end_pos_swing_offset_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "lift_foot", lift_foot_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "swing_foot", swing_foot_);
  }

    catch (std::runtime_error &e) {
    std::cerr << "Error reading parameter [ " << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}