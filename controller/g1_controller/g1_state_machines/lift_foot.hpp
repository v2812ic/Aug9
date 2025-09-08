#pragma once
#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

class LiftFoot : public StateMachine {
public:
  LiftFoot(const StateId state_id, PinocchioRobotSystem *robot,
                       G1ControlArchitecture *ctrl_arch);
  ~LiftFoot() = default;

  void FirstVisit() override;
  void OneStep() override;
  bool EndOfState() override;
  void LastVisit() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  // set nominal desired position/orientation (e.g., for zero acceleration cmd)
  Eigen::Isometry3d nominal_lfoot_iso_;
  Eigen::Isometry3d nominal_rfoot_iso_;

  double time_to_execute_;
  double sway_time_;
  double swing_time_;

  int lift_foot_;
  Eigen::Vector3d foot_offset_;
  Eigen::Vector2d com_offset_;
  double swing_height_;
  bool hold_foot_up_;

  Eigen::Vector2d com_xy_ini;
  Eigen::Vector2d com_xy_target;
  bool swing_is_left_;

  double swing_ref_;

  Eigen::Isometry3d swing_fin_iso_;

};
