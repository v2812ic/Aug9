#pragma once
#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

class MoveFootForward : public StateMachine {
public:
   MoveFootForward(const StateId state_id, PinocchioRobotSystem *robot,
                       G1ControlArchitecture *ctrl_arch);
  ~MoveFootForward() = default;

  void FirstVisit() override;
  void OneStep() override;
  bool EndOfState() override;
  void LastVisit() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  Eigen::Isometry3d nominal_lfoot_iso_;
  Eigen::Isometry3d nominal_rfoot_iso_;

  bool first_time_{true};
  double swing_semitime_;
  double wait_time_;
  Eigen::Vector3d foot_init_pos_swing_offset_;
  Eigen::Vector3d foot_end_pos_swing_offset_;

  double swing_ref_;

  int swing_is_left_;
  int lift_foot_;

  bool swing_foot_;
};

