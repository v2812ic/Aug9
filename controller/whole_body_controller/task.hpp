#pragma once
#include <Eigen/Dense>
#include <iostream>

#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "util/util.hpp"
#include "util/csv_writer.hpp" 

enum class WBC_TYPE { IHWBC, WBIC };

class Task {
public:
  Task(PinocchioRobotSystem *robot, const int dim)
      : robot_(robot), dim_(dim), target_idx_(0),
        local_R_world_(Eigen::Matrix3d::Identity()) {

    des_pos_ = Eigen::VectorXd::Zero(dim_);
    des_vel_ = Eigen::VectorXd::Zero(dim_);
    des_acc_ = Eigen::VectorXd::Zero(dim_);

    local_des_pos_ = Eigen::VectorXd::Zero(dim_);
    local_des_vel_ = Eigen::VectorXd::Zero(dim_);
    local_des_acc_ = Eigen::VectorXd::Zero(dim_);

    pos_ = Eigen::VectorXd::Zero(dim_);
    vel_ = Eigen::VectorXd::Zero(dim_);

    local_pos_ = Eigen::VectorXd::Zero(dim_);
    local_vel_ = Eigen::VectorXd::Zero(dim_);

    pos_err_ = Eigen::VectorXd::Zero(dim_);
    vel_err_ = Eigen::VectorXd::Zero(dim_);

    local_pos_err_ = Eigen::VectorXd::Zero(dim_);
    local_vel_err_ = Eigen::VectorXd::Zero(dim_);

    kp_ = Eigen::VectorXd::Zero(dim_);
    kd_ = Eigen::VectorXd::Zero(dim_);
    ki_ = Eigen::VectorXd::Zero(dim_);

    kp_ik_ = Eigen::VectorXd::Zero(dim_);

    op_cmd_ = Eigen::VectorXd::Zero(dim_);

    jacobian_ = Eigen::MatrixXd::Zero(dim_, robot_->NumQdot());
    jacobian_dot_q_dot_ = Eigen::VectorXd::Zero(dim_);

    weight_ = Eigen::VectorXd::Zero(dim_);
  }
  virtual ~Task() = default;

  // for orientation task, des_pos is a 4 dimensional vector [x,y,z,w]
  // for angular momentum task, des_pos is ignored
  void UpdateDesired(const Eigen::VectorXd &des_pos,
                     const Eigen::VectorXd &des_vel,
                     const Eigen::VectorXd &des_acc) {
    des_pos_ = des_pos;
    des_vel_ = des_vel;
    des_acc_ = des_acc;
  }

  virtual void UpdateOpCommand(
      const Eigen::Matrix3d &world_R_local = Eigen::Matrix3d::Identity()) = 0;
  virtual void UpdateJacobian() = 0;
  virtual void UpdateJacobianDotQdot() = 0;

  // setter function
  // task gain, hierarchy
  virtual void SetParameters(const YAML::Node &node, const WBC_TYPE wbc_type) {
    try {
      util::ReadParameter(node, "kp", kp_);
      util::ReadParameter(node, "kd", kd_);
      if (wbc_type == WBC_TYPE::IHWBC)
        util::ReadParameter(node, "weight", weight_);
      else if (wbc_type == WBC_TYPE::WBIC)
        util::ReadParameter(node, "kp_ik", kp_ik_);
    } catch (std::runtime_error &e) {
      std::cerr << "Error reading parameter [" << e.what() << "] at file: ["
                << __FILE__ << "]" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  void ModifyJacobian(const std::vector<int> joint_idx,
                      const int num_float = 0) {
    for (int i(0); i < joint_idx.size(); i++)
      jacobian_.col(num_float + joint_idx[i]).setZero();
  }

  // getter function
  Eigen::VectorXd DesiredPos() const { return des_pos_; }
  Eigen::VectorXd DesiredVel() const { return des_vel_; }
  Eigen::VectorXd DesiredAcc() const { return des_acc_; }

  Eigen::VectorXd DesiredLocalPos() const { return local_des_pos_; }
  Eigen::VectorXd DesiredLocalVel() const { return local_des_vel_; }
  Eigen::VectorXd DesiredLocalAcc() const { return local_des_acc_; }

  Eigen::VectorXd CurrentPos() const { return pos_; }
  Eigen::VectorXd CurrentVel() const { return vel_; }

  Eigen::VectorXd CurrentLocalPos() const { return local_pos_; }
  Eigen::VectorXd CurrentLocalVel() const { return local_vel_; }

  Eigen::MatrixXd Jacobian() const { return jacobian_; }
  Eigen::MatrixXd JacobianDotQdot() const { return jacobian_dot_q_dot_; }
  Eigen::VectorXd Weight() const { return weight_; }
  Eigen::VectorXd Kp() const { return kp_; }
  Eigen::VectorXd Kd() const { return kd_; }
  Eigen::VectorXd Ki() const { return ki_; }
  Eigen::VectorXd KpIK() const { return kp_ik_; }
  Eigen::VectorXd OpCommand() const { return op_cmd_; }
  int Dim() const { return dim_; }

  Eigen::VectorXd PosError() const { return pos_err_; }
  Eigen::VectorXd LocalPosError() const { return local_pos_err_; }

  int TargetIdx() const { return target_idx_; }
  Eigen::Matrix3d Rot() const { return local_R_world_; }

  // setter
  void SetWeight(Eigen::VectorXd weight) { weight_ = weight; }
  void SetKp(Eigen::VectorXd kp) { kp_ = kp; }
  void SetKd(Eigen::VectorXd kd) { kd_ = kd; }
  void SetKi(Eigen::VectorXd ki) { ki_ = ki; }

  virtual void SetExternalTorque(const Eigen::VectorXd&){}
  virtual void SetExternalForce(const Eigen::VectorXd&){}


  // Debug
  void Debug() {
    std::cout << "=================================" << std::endl;
    std::cout << "des_xddot: " << op_cmd_.transpose() << std::endl;
    std::cout << "pos_err: " << pos_err_.transpose() << std::endl;
    std::cout << "des_pos: " << des_pos_.transpose() << std::endl;
    std::cout << "pos: " << pos_.transpose() << std::endl;
    std::cout << "vel_err: " << vel_err_.transpose() << std::endl;
    std::cout << "des_vel: " << des_vel_.transpose() << std::endl;
    std::cout << "vel: " << vel_.transpose() << std::endl;
  }

  void Export(const std::string& task_str, double t) {
    const std::string base = "experiment_data/tasks/" + task_str;

    // meta.json (solo se crea/actualiza si no existe)
    const std::string meta_path = base + "/meta.json";
    if (!std::filesystem::exists(meta_path)) {
      const int dim = dim_;
      const int nqdot = robot_ ? robot_->NumQdot() : -1;
      std::string meta = "{\n";
      meta += "  \"task\": \"" + task_str + "\",\n";
      meta += "  \"dim\": " + std::to_string(dim) + ",\n";
      meta += "  \"num_qdot\": " + std::to_string(nqdot) + "\n";
      meta += "}\n";
      io::CsvWriter::write_text(meta_path, meta);
    }

    // Helper lambdas para vector/matriz
    auto vec_header = [](const std::string& name, int n) {
      std::vector<std::string> h; h.reserve(n + 1);
      h.push_back("t");
      for (int i = 0; i < n; ++i) h.push_back(name + "_" + std::to_string(i));
      return h;
    };
    auto vec_row = [&](const Eigen::VectorXd& v) {
      std::vector<double> r; r.reserve(v.size() + 1);
      r.push_back(t);
      for (int i = 0; i < v.size(); ++i) r.push_back(v[i]);
      return r;
    };
    auto mat_header = [](const std::string& name, int r, int c) {
      std::vector<std::string> h; h.reserve(r * c + 1);
      h.push_back("t");
      for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
          h.push_back(name + "_" + std::to_string(i) + "_" + std::to_string(j));
        }
      }
      return h;
    };
    auto mat_row = [&](const Eigen::MatrixXd& M) {
      std::vector<double> r; r.reserve(M.rows() * M.cols() + 1);
      r.push_back(t);
      for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
          r.push_back(M(i, j));
      return r;
    };

    // — Señales vectoriales (una fila por ciclo) —
    io::CsvWriter::append_row(base + "/des_pos.csv",  vec_header("des_pos",  des_pos_.size()),  vec_row(des_pos_));
    io::CsvWriter::append_row(base + "/des_vel.csv",  vec_header("des_vel",  des_vel_.size()),  vec_row(des_vel_));
    io::CsvWriter::append_row(base + "/des_acc.csv",  vec_header("des_acc",  des_acc_.size()),  vec_row(des_acc_));

    io::CsvWriter::append_row(base + "/local_des_pos.csv", vec_header("local_des_pos",  local_des_pos_.size()), vec_row(local_des_pos_));
    io::CsvWriter::append_row(base + "/local_des_vel.csv", vec_header("local_des_vel",  local_des_vel_.size()), vec_row(local_des_vel_));
    io::CsvWriter::append_row(base + "/local_des_acc.csv", vec_header("local_des_acc",  local_des_acc_.size()), vec_row(local_des_acc_));

    //io::CsvWriter::append_row(base + "/pos.csv",      vec_header("pos",      pos_.size()),      vec_row(pos_));
    //io::CsvWriter::append_row(base + "/vel.csv",      vec_header("vel",      vel_.size()),      vec_row(vel_));

    //io::CsvWriter::append_row(base + "/local_pos.csv", vec_header("local_pos", local_pos_.size()), vec_row(local_pos_));
    //io::CsvWriter::append_row(base + "/local_vel.csv", vec_header("local_vel", local_vel_.size()), vec_row(local_vel_));

    io::CsvWriter::append_row(base + "/pos_err.csv",  vec_header("pos_err",  pos_err_.size()),  vec_row(pos_err_));
    io::CsvWriter::append_row(base + "/vel_err.csv",  vec_header("vel_err",  vel_err_.size()),  vec_row(vel_err_));

    //io::CsvWriter::append_row(base + "/local_pos_err.csv", vec_header("local_pos_err", local_pos_err_.size()), vec_row(local_pos_err_));
    //io::CsvWriter::append_row(base + "/local_vel_err.csv", vec_header("local_vel_err", local_vel_err_.size()), vec_row(local_vel_err_));

    //io::CsvWriter::append_row(base + "/kp.csv",       vec_header("kp",       kp_.size()),       vec_row(kp_));
    //io::CsvWriter::append_row(base + "/kd.csv",       vec_header("kd",       kd_.size()),       vec_row(kd_));
    //io::CsvWriter::append_row(base + "/ki.csv",       vec_header("ki",       ki_.size()),       vec_row(ki_));
    //io::CsvWriter::append_row(base + "/kp_ik.csv",    vec_header("kp_ik",    kp_ik_.size()),    vec_row(kp_ik_));
    //io::CsvWriter::append_row(base + "/weight.csv",   vec_header("weight",   weight_.size()),   vec_row(weight_));

    //io::CsvWriter::append_row(base + "/op_cmd.csv",   vec_header("op_cmd",   op_cmd_.size()),   vec_row(op_cmd_));

    // — Señales matriciales —
    //io::CsvWriter::append_row(base + "/jacobian.csv",
                              //mat_header("J", jacobian_.rows(), jacobian_.cols()),
                              //mat_row(jacobian_));
    //io::CsvWriter::append_row(base + "/jacobian_dot_q_dot.csv",
                              //vec_header("Jdot_qdot", jacobian_dot_q_dot_.size()),
                              //vec_row(jacobian_dot_q_dot_));
  }

protected:
  PinocchioRobotSystem *robot_;
  int dim_;
  int target_idx_;

  Eigen::Matrix3d local_R_world_;

  // measured quantities
  Eigen::VectorXd pos_;
  Eigen::VectorXd vel_;

  Eigen::VectorXd local_pos_;
  Eigen::VectorXd local_vel_;

  Eigen::VectorXd pos_err_;
  Eigen::VectorXd vel_err_;

  Eigen::VectorXd local_pos_err_;
  Eigen::VectorXd local_vel_err_;

  // task space gains
  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
  Eigen::VectorXd ki_;

  // ik gains
  Eigen::VectorXd kp_ik_;

  //  desired quantities
  Eigen::VectorXd des_pos_;
  Eigen::VectorXd des_vel_;
  Eigen::VectorXd des_acc_;

  Eigen::VectorXd local_des_pos_;
  Eigen::VectorXd local_des_vel_;
  Eigen::VectorXd local_des_acc_;

  Eigen::VectorXd op_cmd_;

  Eigen::MatrixXd jacobian_;
  Eigen::VectorXd jacobian_dot_q_dot_;

  Eigen::VectorXd weight_;
};
