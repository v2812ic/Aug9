#include "controller/g1_controller/g1_state_machines/lift_foot.hpp"
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

// --- FUNCIÓN AUXILIAR PARA APLANAR LA ORIENTACIÓN DEL PIE ---
// Devuelve una pose con la misma posición pero con orientación neutra (horizontal).
namespace {
Eigen::Isometry3d MakeHorizontal(const Eigen::Isometry3d& iso) {
    Eigen::Isometry3d new_iso = Eigen::Isometry3d::Identity();
    new_iso.translation() = iso.translation();
    return new_iso;
}
}

LiftFoot::LiftFoot(const StateId state_id,
                                           PinocchioRobotSystem *robot,
                                           G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch) {
  util::PrettyConstructor(2, "Lifting Foot (Stable Version)");
  nominal_lfoot_iso_.setIdentity();
  nominal_rfoot_iso_.setIdentity();
  sp_ = G1StateProvider::GetStateProvider();
}

void LiftFoot::FirstVisit() {
  std::cout << "g1_states: Lifting Foot" << std::endl;
  state_machine_start_time_ = sp_->current_time_;
  end_time_ = state_machine_start_time_ + time_to_execute_;
  swing_time_ = time_to_execute_ - sway_time_;

  swing_is_left_ = (lift_foot_ == 0);
  if (swing_is_left_){
    sp_->b_lf_contact_ = false;
    sp_->b_rf_contact_ = true;
    sp_->b_swing_leg_ = end_effector::LFoot;
  } else {
    sp_->b_lf_contact_ = true;
    sp_->b_rf_contact_ = false;
    sp_->b_swing_leg_ = end_effector::RFoot;
  }

  auto com_xy_task = ctrl_arch_->tci_container_->task_map_["com_xy_task"];
  com_xy_ini = com_xy_task->DesiredPos();
  
  if (swing_is_left_) {
    Eigen::Isometry3d stance_foot_iso = robot_->GetLinkIsometry(g1_link::r_foot_contact);
    com_xy_target = stance_foot_iso.translation().head<2>();
    // Incluir rampa de fuerza
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
  //com_xy_target = com_xy_ini;
  com_xy_target += com_offset_;

  if (swing_is_left_) {
    Eigen::Isometry3d start_iso = MakeHorizontal(robot_->GetLinkIsometry(g1_link::l_foot_contact));
    Eigen::Isometry3d final_iso = start_iso;
    final_iso.translation() += foot_offset_;
    const double apex_z = start_iso.translation().z() + swing_height_;

    ctrl_arch_->lf_SE3_tm_->InitializeSwingTrajectory(start_iso, final_iso, apex_z, swing_time_);

    swing_fin_iso_ = final_iso;
    nominal_rfoot_iso_ = robot_->GetLinkIsometry(g1_link::r_foot_contact);
  } else {
    Eigen::Isometry3d start_iso = MakeHorizontal(robot_->GetLinkIsometry(g1_link::r_foot_contact));
    Eigen::Isometry3d final_iso = start_iso;
    final_iso.translation() += foot_offset_;
    const double apex_z = start_iso.translation().z() + swing_height_;

    ctrl_arch_->rf_SE3_tm_->InitializeSwingTrajectory(start_iso, final_iso, apex_z, swing_time_);

    swing_fin_iso_ = final_iso;
    nominal_lfoot_iso_ = robot_->GetLinkIsometry(g1_link::l_foot_contact);
  }
}

void LiftFoot::OneStep() {
  state_machine_time_ = sp_->current_time_ - state_machine_start_time_;
  swing_ref_ = std::max(0.0, sp_->current_time_ - state_machine_start_time_ - sway_time_ - 1);

  // 1) Interpolar CoM
  double s = util::SmoothPos(0, 1, sway_time_, state_machine_time_);
  s = std::clamp(s, 0.0, 1.0);
  Eigen::Vector2d com_xy_des = (1.0 - s) * com_xy_ini + s * com_xy_target;
  ctrl_arch_->tci_container_->task_map_["com_xy_task"]
      ->UpdateDesired(com_xy_des, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

  // 2) Actualizar pies si cal
  if (swing_is_left_) {
    ctrl_arch_->lf_SE3_tm_->UpdateDesired(swing_ref_);
    ctrl_arch_->rf_SE3_tm_->UseNominal(nominal_rfoot_iso_);
  } else {
    ctrl_arch_->rf_SE3_tm_->UpdateDesired(swing_ref_);
    ctrl_arch_->lf_SE3_tm_->UseNominal(nominal_lfoot_iso_);
  }

  if (hold_foot_up_ && state_machine_time_ >= time_to_execute_) {
    if (swing_is_left_) {
      ctrl_arch_->lf_SE3_tm_->UseNominal(swing_fin_iso_);
    } else {
      ctrl_arch_->rf_SE3_tm_->UseNominal(swing_fin_iso_);
    }
  }

  // 3) Actualizar fuerzas de contanto
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
}

bool LiftFoot::EndOfState() { 
  if (hold_foot_up_) return false;
  return (state_machine_time_ >= end_time_);
}

void LiftFoot::LastVisit() {
  if (!hold_foot_up_) { 
    sp_->b_lf_contact_ = true; 
    sp_->b_rf_contact_ = true; 
  }
}

StateId LiftFoot::GetNextState() {
    return g1_states::kDoubleSupportBalance;
}

void LiftFoot::SetParameters(const YAML::Node &node) {
  try {
    util::ReadParameter(node["state_machine"]["lifting_foot"], "foot_offset", foot_offset_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "com_xy_offset", com_offset_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "lift_foot", lift_foot_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "time_to_execute", time_to_execute_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "sway_time", sway_time_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "hold_foot_up", hold_foot_up_);
    util::ReadParameter(node["state_machine"]["lifting_foot"], "swing_height", swing_height_);
  } catch (std::runtime_error &e) {
    std::cerr << "Error reading parameter [ " << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}