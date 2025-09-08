#include "controller/whole_body_controller/ihwbc/ihwbc.hpp"
#include "controller/whole_body_controller/contact.hpp"

#include "controller/whole_body_controller/force_task.hpp"
#include "controller/whole_body_controller/internal_constraint.hpp"
#include "controller/whole_body_controller/task.hpp"

IHWBC::IHWBC(const std::vector<bool> &act_qdot_list)
    : WBC(act_qdot_list), dim_cone_constraint_(0), lambda_qddot_(0.),
      b_first_visit_(true) {
  util::PrettyConstructor(3, "IHWBC");

  // assume surface contact:TODO make generic
  // lambda_rf_ = Eigen::VectorXd::Ones(12);

  tau_ext_.setZero();
  f_ext_.setZero();
}

void IHWBC::Solve(const std::unordered_map<std::string, Task *> &task_map,
                  const std::map<std::string, Contact *> &contact_map,
                  const std::unordered_map<std::string, InternalConstraint *>
                      &internal_constraint_map,
                  std::map<std::string, ForceTask *> &force_task_map,
                  Eigen::VectorXd &qddot_cmd, Eigen::VectorXd &trq_cmd) {

  bool kPrint_ = ((iter_++ % print_every_) == 0);

  assert(task_map.size() > 0);
  b_contact_ = contact_map.size() > 0 ? true : false;

  // AHORA ESTAMOS TOCANDO AQUI JEFES ===========================
  //=============================================================
  // cost setup
  //=============================================================
  Eigen::MatrixXd cost_mat, cost_t_mat, cost_rf_mat;
  Eigen::VectorXd cost_vec, cost_t_vec, cost_rf_vec;

  
  // task cost
  cost_t_mat.setZero(num_qdot_, num_qdot_);
  cost_t_vec.setZero(num_qdot_);
  for (const auto &[task_str, task_ptr] : task_map) {
    Eigen::MatrixXd jt = task_ptr->Jacobian();
    Eigen::VectorXd jtdot_qdot = task_ptr->JacobianDotQdot();
    Eigen::VectorXd des_xddot = task_ptr->OpCommand();
    Eigen::MatrixXd weight_mat = task_ptr->Weight().asDiagonal();

    //std::cout << task_str << std::endl;
    // std::cout
    //<< "--------------------------------------------------------------"
    //<< std::endl;
    // std::cout << "jt" << std::endl;
    // std::cout << jt << std::endl;

    // Debug Task
    // task_ptr->Debug();

    if (kPrint_){
      //std::cout << "[IHWBC] weight task \"" << task_str << "\": " << jt.transpose() * weight_mat * jt << " | " << (jtdot_qdot - des_xddot).transpose() * weight_mat * jt << std::endl;
    }

    cost_t_mat += jt.transpose() * weight_mat * jt;
    cost_t_vec += (jtdot_qdot - des_xddot).transpose() * weight_mat * jt;
  }
  cost_t_mat += lambda_qddot_ * M_; // regularization term

  // // check contact dimension. Es solo para setear dimensiones, da igual
  if (b_first_visit_) {
    for (const auto &[contact_str, contact_ptr] : contact_map) {
      dim_contact_ += contact_ptr->Dim();
      dim_cone_constraint_ += contact_ptr->UfVector().size();
      b_first_visit_ = false;
    }
  }

  cost_mat.setZero(num_qdot_ + dim_contact_, num_qdot_ + dim_contact_);
  cost_vec.setZero(num_qdot_ + dim_contact_);

  // reaction force cost
  if (b_contact_) {
    // exist contact
    cost_rf_mat.setZero(dim_contact_, dim_contact_);
    cost_rf_vec.setZero(dim_contact_);
    int row_idx(0);
    for (const auto &[force_task_str, force_task_ptr] : force_task_map) {
      // lfoot, rfoot order & wrench (torque, force order)
      Eigen::MatrixXd weight_mat = force_task_ptr->Weight().asDiagonal();
      Eigen::VectorXd des_rf = force_task_ptr->DesiredRf();
      int dim = force_task_ptr->Dim();

      cost_rf_mat.block(row_idx, row_idx, dim, dim) = weight_mat;
      cost_rf_vec.segment(row_idx, dim) = -des_rf.transpose() * weight_mat;
      row_idx += dim;
    }
    cost_rf_mat += lambda_rf_.asDiagonal();
    // util::PrettyPrint(cost_rf_mat, std::cout, "cost_rf_mat");

    cost_mat.topLeftCorner(cost_t_mat.rows(), cost_t_mat.cols()) = cost_t_mat;
    cost_mat.bottomRightCorner(cost_rf_mat.rows(), cost_rf_mat.cols()) =
        cost_rf_mat;
    cost_vec.head(cost_t_vec.size()) = cost_t_vec;
    cost_vec.tail(cost_rf_vec.size()) = cost_rf_vec;
  } else {
    // no contact
    cost_mat = cost_t_mat;
    cost_vec = cost_t_vec;
  }

  // HASTA AQUI, EN ADELANTE VAMOS CHETADOS Y LO HEMOS EDITADO BIEN (AL MENOS PARA EL G1)
  //=============================================================
  // equality constraint (Dynamics) setup
  //=============================================================

  // internal constraints setup - inutil para el caso G1
  Eigen::MatrixXd ji = Eigen::MatrixXd::Zero(num_passive_, num_qdot_);
  Eigen::VectorXd jidot_qdot_vec = Eigen::VectorXd(num_passive_);
  Eigen::VectorXd ji_transpose_lambda_int_jidot_qdot_vec =
      Eigen::VectorXd::Zero(num_qdot_);

  Eigen::MatrixXd ni;
  Eigen::MatrixXd sa_ni_trc_bar; // TODO: sa_ni_trc_bar using just pseudo inv
                                 // (not dynamically consistent pseudo inv)
  if (b_internal_constraint_) {
    // exist passive joint
    int row_idx(0);
    for (const auto [ic_str, ic_ptr] : internal_constraint_map) {
      Eigen::MatrixXd j_i = ic_ptr->Jacobian();
      Eigen::VectorXd jidot_qdot = ic_ptr->JacobianDotQdot();
      int dim = ic_ptr->Dim();

      ji.middleRows(row_idx, dim) = j_i;
      jidot_qdot_vec.segment(row_idx, dim) = jidot_qdot;
      row_idx += dim;
    }

    Eigen::MatrixXd lambda_int_inv = ji * Minv_ * ji.transpose();
    ji_transpose_lambda_int_jidot_qdot_vec =
        ji.transpose() * util::PseudoInverse(lambda_int_inv, 0.0001) *
        jidot_qdot_vec;

    Eigen::MatrixXd ji_bar = util::WeightedPseudoInverse(ji, Minv_, 0.0001);
    ni = Eigen::MatrixXd::Identity(num_qdot_, num_qdot_) - ji_bar * ji;

    // compuationally efficient pseudo inverse for trq calc (exclude floating
    // base)
    Eigen::MatrixXd sa_ni_trc =
        (sa_ * ni).rightCols(num_active_ + num_passive_);
    Eigen::MatrixXd Ainv_trc = Minv_.bottomRightCorner(
        num_active_ + num_passive_, num_active_ + num_passive_);
    sa_ni_trc_bar = util::WeightedPseudoInverse(sa_ni_trc, Ainv_trc, 0.00001);
    // sa_ni_trc_bar = util::PseudoInverse(
    // sa_ni_trc,
    // 0.0001); // TODO: only apply for Draco3 (due to internal constraint jac)

    // TEST TODO
    // Eigen::MatrixXd sa_ni = sa_ * ni;
    // Eigen::MatrixXd sa_ni_bar =
    // util::WeightedPseudoInverse(sa_ni, Minv_, 0.00001);
    // std::cout << "---------------test result:----------------------- "
    //<< std::endl;
    // std::cout << sa_ni_bar * sa_ni - ni << std::endl;
    // std::cout << "=============sa_ni_bar * sa_ni==================="
    //<< std::endl;
    // std::cout << sa_ni_bar * sa_ni << std::endl;
    // std::cout << "==================ni============================"
    //<< std::endl;
    // std::cout << ni << std::endl;
    // std::cout << sf_ * grav_ << std::endl;

  } else {
    // no passive joint
    ni.setIdentity(num_qdot_, num_qdot_);
    ji_transpose_lambda_int_jidot_qdot_vec.setZero(num_qdot_);
    sa_ni_trc_bar.setIdentity(num_active_, num_active_);
  }

  // contact setup
  Eigen::MatrixXd jc, uf_mat;
  Eigen::VectorXd uf_vec;
  if (b_contact_) {
    // contact exist
    jc.setZero(dim_contact_, num_qdot_);
    uf_mat.setZero(dim_cone_constraint_, dim_contact_);
    uf_vec.setZero(dim_cone_constraint_);

    int contact_row_idx(0);
    int contact_cone_row_idx(0);
    for (const auto &[contact_str, contact_ptr] : contact_map) {
      Eigen::MatrixXd j_c = contact_ptr->Jacobian();
      Eigen::MatrixXd cone_mat = contact_ptr->UfMatrix();
      Eigen::VectorXd cone_vec = contact_ptr->UfVector();
      int dim_contact = contact_ptr->Dim();
      int dim_cone_constraint = cone_mat.rows();

      jc.middleRows(contact_row_idx, dim_contact) = j_c;
      uf_mat.block(contact_cone_row_idx, contact_row_idx, dim_cone_constraint,
                   dim_contact) = cone_mat;
      uf_vec.segment(contact_cone_row_idx, dim_cone_constraint) = cone_vec;
      contact_row_idx += dim_contact;
      contact_cone_row_idx += dim_cone_constraint;

      //contact_ptr->Print();
    }
  } else {
    // no contact
  }

  //util::PrettyPrint(uf_mat, std::cout, "uf_mat");
  //util::PrettyPrint(uf_vec, std::cout, "uf_vec");

  // { HECHO EL CÁLCULO DE LAS MATRICES DE IGUALDAD}
  // equality constraints mat & vec 
  Eigen::MatrixXd eq_mat, eq_float_mat, eq_int_mat;
  Eigen::VectorXd eq_vec, eq_float_vec, eq_int_vec;

  Eigen::MatrixXd h = cori_ + grav_;
  if (tau_ext_.size() > 0){
    h -= tau_ext_;
  }

  if (b_contact_) { // CONTACTOS ACTIVOS CONTRA OTRAS SUPERFICIES
    if (b_floating_base_) { // EL ROBOT NO ESTA ANCLADO; ES FLOATING BASE (Lo que nos interesa)
      if (b_internal_constraint_) { // El robot tiene restricciones en algunas juntas (no nos interesa)
        // floating base: o, internal constraint: o, contact: o
        eq_float_mat.setZero(num_floating_, num_qdot_ + dim_contact_);
        eq_float_vec.setZero(num_floating_);

        eq_float_mat.leftCols(num_qdot_) = sf_ * M_;
        eq_float_mat.rightCols(dim_contact_) =
            -sf_ * ni.transpose() * jc.transpose();
        eq_float_vec = sf_ * ni.transpose() * h;

        eq_int_mat.setZero(num_passive_, num_qdot_ + dim_contact_);
        eq_int_vec.setZero(num_passive_);

        eq_int_mat.leftCols(num_qdot_) = ji;
        eq_int_vec = jidot_qdot_vec;

        eq_mat.setZero(num_floating_ + num_passive_, num_qdot_ + dim_contact_);
        eq_vec.setZero(num_floating_ + num_passive_);

        eq_mat.topRows(num_floating_) = eq_float_mat;
        eq_mat.bottomRows(num_passive_) = eq_int_mat;
        eq_vec.head(num_floating_) = eq_float_vec;
        eq_vec.tail(num_passive_) = eq_int_vec;

        // std::cout <<
        // "======================================================="
        //<< std::endl;
        // std::cout << "eq_mat" << std::endl;
        // std::cout << eq_mat << std::endl;
        // std::exit(0);

      } else { // NOS INTERESA --------------------------------------------------------------<
        // floating base: o, internal constraint: x, contact: o
        //-----------------------------------------------------------------

        // ESTE ES EL PRINCIPAL EN EL QUE VAMOS A ENTRAR SIEMPRE AL MENOS EN ESTE TFG Y CON EL G1

        //-----------------------------------------------------------------
        eq_float_mat.setZero(num_floating_, num_qdot_ + dim_contact_);
        eq_float_vec.setZero(num_floating_);

        eq_float_mat.leftCols(num_qdot_) = sf_ * M_;
        eq_float_mat.rightCols(dim_contact_) =
            -sf_ * ni.transpose() * jc.transpose();

        eq_float_vec = sf_ * ni.transpose() * h;

        //util::PrettyPrint(ni, std::cout, "ni");
        //util::PrettyPrint(sf_, std::cout, "sf_");

        eq_mat = eq_float_mat;
        eq_vec = eq_float_vec;

      }
    } else { // NO NOS INTERESA
      if (b_internal_constraint_) {
        // floating base: x, internal constraint: o, contact: o
        eq_int_mat.setZero(num_passive_, num_qdot_ + dim_contact_);
        eq_int_mat.leftCols(num_qdot_) = ji;
        eq_int_vec = jidot_qdot_vec;

        eq_mat = eq_int_mat;
        eq_vec = eq_int_vec;

      } else {
        // floating base: x, internal constraint: x, contact: o
        eq_mat.setZero(0, num_qdot_ + dim_contact_);
        eq_vec.setZero(0);
      }
    }
  } else { // NO HAY CONTACTOS ACTIVOS, EN PRINCIPIO ESTO NO NOS INTERESA A NO SER QUE VUELE
    if (b_floating_base_) { // NOS INTERESA
      if (b_internal_constraint_) { // NO NOS INTERESA
        // floating base: o, internal constraint: o, contact: x
        eq_float_mat = sf_ * M_;
        eq_float_vec = sf_ * ni.transpose() * h;

        eq_int_mat = ji;
        eq_int_vec = jidot_qdot_vec;

        eq_mat.setZero(num_floating_ + num_passive_, num_qdot_);
        eq_vec.setZero(num_floating_ + num_passive_);

        eq_mat.topRows(num_floating_) = eq_float_mat;
        eq_mat.bottomRows(num_passive_) = eq_int_mat;
        eq_vec.head(num_floating_) = eq_float_vec;
        eq_vec.tail(num_passive_) = eq_int_vec;
      } else { // NOS INTERESA --------------------------------------------------------------<
        // floating base: o, internal contstraint: x, contact: x
        eq_float_mat = sf_ * M_;
        eq_float_vec = sf_ * ni.transpose() * h;

        eq_mat = eq_float_mat;
        eq_vec = eq_float_vec;
      }
    } else {
      if (b_internal_constraint_) {
        // floating base: x, internal constraint: o, contact: x
        eq_int_mat = ji;
        eq_int_vec = jidot_qdot_vec;

        eq_mat = eq_int_mat;
        eq_vec = eq_int_vec;

      } else {
        // floating base: x, internal constraint: x, contact: x
        eq_mat.setZero(0, num_qdot_);
        eq_vec.setZero(0);
      }
    }
  }

  //=============================================================
  // inequality constraint setup
  //=============================================================
  // Uf >= contact_cone_vec
  Eigen::MatrixXd ineq_mat;
  Eigen::VectorXd ineq_vec;


  if (!b_trq_limit_) {
    if (b_contact_) {
      // trq limit : x, contact: o 
      ineq_mat.setZero(dim_cone_constraint_, num_qdot_ + dim_contact_);
      ineq_vec.setZero(dim_cone_constraint_);

      ineq_mat.rightCols(dim_contact_) = uf_mat;
      ineq_vec = -uf_vec;

    } else {
      // trq limit: x, contact: x
      ineq_mat.setZero(0, num_qdot_);
      ineq_vec.setZero(0);
    }
  } else { // AQUI, TRQ LIMIT ON, B_CONTACT ON
    Eigen::MatrixXd ineq_trq_mat;
    Eigen::VectorXd ineq_trq_vec;

    //std::cout << "dim_cone_constraint_: "  << dim_cone_constraint_ << std::endl;
    //std::cout << "num_qdot_: "  << num_qdot_ << std::endl;
    //std::cout << "dim_contact_: "  << dim_contact_ << std::endl;

    Eigen::MatrixXd l_trq_mat, r_trq_mat;
    Eigen::VectorXd l_trq_vec, r_trq_vec;
    l_trq_mat.setZero(num_qdot_ - num_floating_, num_qdot_ + dim_contact_);
    r_trq_mat.setZero(num_qdot_ - num_floating_, num_qdot_ + dim_contact_);
    // l_trq_vec.setZero(num_qdot_ - num_floating_);
    // r_trq_vec.setZero(num_qdot_ - num_floating_);

    l_trq_mat.leftCols(num_qdot_) = sa_ni_trc_bar.transpose() * snf_ * M_;
    
    l_trq_mat.rightCols(dim_contact_) =
        -sa_ni_trc_bar.transpose() * snf_ * (jc * ni).transpose();
    l_trq_vec =
        - joint_trq_limits_.leftCols(1) +
        sa_ni_trc_bar.transpose() * snf_ * ni.transpose() * h +
        sa_ni_trc_bar.transpose() * snf_ *
            ji_transpose_lambda_int_jidot_qdot_vec;

    r_trq_mat = -l_trq_mat;
    r_trq_vec =
        joint_trq_limits_.rightCols(1) -
        sa_ni_trc_bar.transpose() * snf_ * ni.transpose() * h -
        sa_ni_trc_bar.transpose() * snf_ *
            ji_transpose_lambda_int_jidot_qdot_vec;

    ineq_trq_mat.setZero(2 * (num_qdot_ - num_floating_),
                         num_qdot_ + dim_contact_);
    ineq_trq_vec.setZero(2 * (num_qdot_ - num_floating_));
    ineq_trq_mat.topRows(num_qdot_ - num_floating_) = l_trq_mat;
    ineq_trq_mat.bottomRows(num_qdot_ - num_floating_) = r_trq_mat;
    ineq_trq_vec.head(num_qdot_ - num_floating_) = l_trq_vec;
    ineq_trq_vec.tail(num_qdot_ - num_floating_) = r_trq_vec;

    if (b_contact_) { // Y AQUI
      // trq limit: o, contact: o
      Eigen::MatrixXd ineq_contact_mat;
      Eigen::VectorXd ineq_contact_vec;

      ineq_contact_mat.setZero(dim_cone_constraint_, num_qdot_ + dim_contact_);
      ineq_contact_vec.setZero(dim_cone_constraint_);

      ineq_contact_mat.rightCols(dim_contact_) = uf_mat;
      ineq_contact_vec.tail(dim_cone_constraint_) = -uf_vec;
      // TIENEN SENTIDO

      ineq_mat.setZero(ineq_trq_mat.rows() + ineq_contact_mat.rows(),
                       num_qdot_ + dim_contact_);
      ineq_vec.setZero(ineq_trq_mat.rows() + ineq_contact_mat.rows());

      ineq_mat.topRows(ineq_trq_mat.rows()) = ineq_trq_mat; // ok y dimensiones
      ineq_mat.bottomRows(ineq_contact_mat.rows()) = ineq_contact_mat; // ok
      ineq_vec.head(ineq_trq_mat.rows()) = ineq_trq_vec; // ok y dimensiones
      ineq_vec.tail(ineq_contact_mat.rows()) = ineq_contact_vec; // ok

    } else { // AQUI NOS LA PELA JEFE
      // trq limit: o, contact: x
      ineq_mat = ineq_trq_mat;
      ineq_vec = ineq_trq_vec;
    }
  }

  // set up QP formulation & solve QP
  // quadprog QP formulation template Eq.
  /*
       min
          0.5 * x G x + g0 x
       s.t.
          CE^T x + ce0 = 0
          CI^T x + ci0 >= 0
      */
  num_qp_vars_ = cost_mat.cols();
  num_eq_const_ = eq_mat.rows();
  num_ineq_const_ = ineq_mat.rows();

  qddot_sol_.setZero(num_qdot_);
  rf_sol_.setZero(dim_contact_);

  x_.resize(num_qp_vars_);
  G_.resize(num_qp_vars_, num_qp_vars_);
  g0_.resize(num_qp_vars_);
  CE_.resize(num_qp_vars_, num_eq_const_);
  ce0_.resize(num_eq_const_);
  CI_.resize(num_qp_vars_, num_ineq_const_);
  ci0_.resize(num_ineq_const_);

  _SetQPCost(cost_mat, cost_vec);
  _SetQPEqualityConstraint(eq_mat, eq_vec);
  _SetQPInEqualityConstraint(ineq_mat, ineq_vec);
  _SolveQP();

  // compute torque command
  if (b_contact_) { // ESTE, HAY CONTACTO
    // contact: o
    trq_cmd = sa_.rightCols(num_qdot_ - num_floating_).transpose() *
              sa_ni_trc_bar.transpose() * snf_ *
              (M_ * qddot_sol_ + ni.transpose() * h -
               (jc * ni).transpose() * rf_sol_ +
               ji_transpose_lambda_int_jidot_qdot_vec);
  } else {
    // contact: x
    trq_cmd = sa_.rightCols(num_qdot_ - num_floating_).transpose() *
              sa_ni_trc_bar.transpose() * snf_ *
              (M_ * qddot_sol_ + ni.transpose() * h +
               ji_transpose_lambda_int_jidot_qdot_vec);
  }

  // qddot_cmd = sa_ * qddot_sol_;
  // rf_cmd = rf_sol_;
  qddot_cmd = qddot_sol_;
  // force_task_map["lf_force_task"]->UpdateCmd(rf_sol_.head(dim_contact_ / 2));
  // force_task_map["rf_force_task"]->UpdateCmd(rf_sol_.tail(dim_contact_ / 2));

  // TODO: make it general
  int i(0);
  for (const auto &kv : force_task_map) {
    kv.second->UpdateCmd(rf_sol_.segment(i, dim_contact_ / 2));
    i += dim_contact_ / 2;
  }

  if (num_ineq_const_ > 0) {
    Eigen::VectorXd x_full(num_qdot_ + dim_contact_);
    x_full << qddot_sol_, rf_sol_;

    // CI^T x + ci0 >= 0  con  CI = ineq_mat^T, ci0 = ineq_vec
    Eigen::VectorXd left  = ineq_mat * x_full; // a_i^T x
    Eigen::VectorXd right = ineq_vec;                      // -b_i
    Eigen::VectorXd slack = left + right;                   // = a_i^T x + b_i  (>= 0 ideal)

    int nJ = (b_trq_limit_ ? (num_qdot_ - num_floating_) : 0);
    int trq_rows = (b_trq_limit_ ? 2 * nJ : 0);
    int contact_rows = (b_contact_ ? dim_cone_constraint_ : 0);



    bool printed_ = false;
    
    {/*
    for (int i = 0; i < slack.size(); ++i) {
    
      std::string tag = "ineq_row[" + std::to_string(i) + "]";
      if (b_trq_limit_ && i < trq_rows) {
        if (i < nJ)  tag = "tau_lower[j=" + std::to_string(i) + "]";
        
        else         tag = "tau_upper[j=" + std::to_string(i - nJ) + "]";
        
      } else if (b_contact_ && i >= trq_rows && i < trq_rows + contact_rows) {
        tag = "contact_cone[row=" + std::to_string(i - trq_rows) + "]";
        
      }

      if (slack[i] < 1e-4) {
        std::cout << "[IHWBC][WARNING-CLOSE TO LIMIT] " << tag
                  << "  slack=" << slack[i] << std::endl;
                  printed_ = true;
      } 
    }
    if (printed_){
        std::cout << "==================="<< std::endl;
    }
    */}
  }
}

void IHWBC::ComputeTaskCosts(
    const std::unordered_map<std::string, Task *> &task_map,
    const std::map<std::string, ForceTask *> &force_task_map,
    std::unordered_map<std::string, double> &task_unweighted_cost_map,
    std::unordered_map<std::string, double> &task_weighted_cost_map) {
  if (task_unweighted_cost_map.empty())
    return;

  // save cost associated with each task
  for (const auto &[task_str, task_ptr] : task_map) {
    Eigen::MatrixXd jac_t = task_ptr->Jacobian();
    Eigen::MatrixXd jacdot_t_qdot = task_ptr->JacobianDotQdot();
    Eigen::VectorXd xddot_des_t = task_ptr->OpCommand();
    Eigen::VectorXd cost_t_sum_vec =
        jac_t * qddot_sol_ + jacdot_t_qdot - xddot_des_t;
    task_unweighted_cost_map[task_str] =
        cost_t_sum_vec.transpose() * cost_t_sum_vec;
    task_weighted_cost_map.at(task_str) = cost_t_sum_vec.transpose() *
                                          task_ptr->Weight().asDiagonal() *
                                          cost_t_sum_vec;
  }
  for (const auto &[force_task_str, force_task_ptr] : force_task_map) {
    Eigen::VectorXd Fr_error =
        force_task_ptr->DesiredRf() - force_task_ptr->CmdRf();
    Eigen::MatrixXd weight_mat = force_task_ptr->Weight().asDiagonal();
    task_unweighted_cost_map[force_task_str] = Fr_error.transpose() * Fr_error;
    task_weighted_cost_map[force_task_str] =
        Fr_error.transpose() * weight_mat * Fr_error;
  }
  if (task_unweighted_cost_map.count("qddot_regularization_task")) {
    task_unweighted_cost_map["qddot_regularization_task"] =
        qddot_sol_.transpose() * qddot_sol_;
  }
  if (task_weighted_cost_map.count("qddot_regularization_task")) {
    task_weighted_cost_map["qddot_regularization_task"] =
        lambda_qddot_ * qddot_sol_.transpose() * qddot_sol_;
  }
  if (task_unweighted_cost_map.count("Fr_regularization_task")) {
    task_unweighted_cost_map["Fr_regularization_task"] =
        rf_sol_.transpose() * rf_sol_;
    task_weighted_cost_map["Fr_regularization_task"] =
        rf_sol_.transpose() * lambda_rf_.asDiagonal() * rf_sol_;
  }
}

void IHWBC::_SetQPCost(const Eigen::MatrixXd &cost_mat,
                       const Eigen::VectorXd &cost_vec) {
  for (int i(0); i < num_qp_vars_; ++i) {
    for (int j(0); j < num_qp_vars_; ++j) {
      G_[i][j] = cost_mat(i, j);
    }
    g0_[i] = cost_vec[i];
  }
  //std::cout << cost_mat << std::endl;
}

void IHWBC::_SetQPEqualityConstraint(const Eigen::MatrixXd &eq_mat,
                                     const Eigen::VectorXd &eq_vec) {
  for (int i(0); i < num_eq_const_; ++i) {
    for (int j(0); j < num_qp_vars_; ++j) {
      CE_[j][i] = eq_mat(i, j);
    }
    ce0_[i] = eq_vec[i];
  }
}

void IHWBC::_SetQPInEqualityConstraint(const Eigen::MatrixXd &ineq_mat,
                                       const Eigen::VectorXd &ineq_vec) {
  for (int i(0); i < num_ineq_const_; ++i) {
    for (int j(0); j < num_qp_vars_; ++j) {
      CI_[j][i] = ineq_mat(i, j);
    }
    ci0_[i] = ineq_vec[i];
  }
}

void IHWBC::_SolveQP() {
  double qp_result = solve_quadprog(G_, g0_, CE_, ce0_, CI_, ci0_, x_);

  Eigen::VectorXd qp_sol = Eigen::VectorXd::Zero(num_qp_vars_);
  for (int i(0); i < num_qp_vars_; ++i)
    qp_sol[i] = x_[i];

  qddot_sol_ = qp_sol.head(num_qdot_);
  rf_sol_ = qp_sol.tail(dim_contact_);
}

void IHWBC::SetParameters(const YAML::Node &node) {
  try {
    util::ReadParameter(node, "lambda_qddot", lambda_qddot_);
    util::ReadParameter(node, "lambda_rf", lambda_rf_);
    util::ReadParameter(node, "b_trq_limit", b_trq_limit_);
  } catch (std::runtime_error &e) {
    std::cerr << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
