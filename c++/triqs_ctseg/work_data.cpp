#include "work_data.hpp"
#include <nda/basic_functions.hpp>
#include <nda/traits.hpp>
#include <triqs/gfs/functions/functions2.hpp>
#include <triqs/operators/util/extractors.hpp>
#include "logs.hpp"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"

// Work data constructor
work_data_t::work_data_t(params_t const &p, inputs_t const &inputs, mpi::communicator c) {

  // Set logger level
  spdlog::set_pattern("%v");
  spdlog::set_level(spdlog::level::info);
  if constexpr (print_logs) spdlog::set_level(spdlog::level::debug);

  // Copy data from inputs
  double beta = p.beta;
  gf_struct   = p.gf_struct;
  n_color     = count_colors(gf_struct);

  // Compute gf_block_size_partial_sum
  { // to make acc local
    long acc = 0;
    for (auto const &[s, l] : gf_struct) {
      if (l > 1) offdiag_delta = true;
      gf_block_size_partial_sum.push_back(acc);
      acc += l;
    }
  }

  // Compute color/block conversion tables
  for (auto const &color : range(n_color)) {
    block_number.push_back(find_block_number(color));
    index_in_block.push_back(find_index_in_block(color));
  }

  // Color-dependent chemical potential
  mu = nda::zeros<double>(n_color);
  if (p.hartree_shift.size() > 0) {
    ALWAYS_EXPECTS((p.hartree_shift.size() == n_color), "Hartree shift size is not {}", n_color);
    mu = p.hartree_shift;
  }

  // .............. Interactions .................
  // Extract the U from the operator
  auto U_full = triqs::operators::utils::dict_to_matrix(triqs::operators::utils::extract_U_dict2(p.h_int), p.gf_struct);
  U           = nda::matrix<double>{real(U_full)};
  // We ensure that U(a, a) is 0, which must be true
  for (int a = 0; a < U.extent(0); ++a) ALWAYS_EXPECTS((abs(U(a, a)) < 1.e-15), "Internal Error. U diag non zero");

  // Do we have D(tau) and J_perp(tau)? Yes, unless the data is 0
  has_Dt    = max_element(abs(inputs.d0t.data())) > 1.e-13;
  has_jperp = max_element(abs(inputs.jperpt.data())) > 1.e-13;

  // Check: no J_perp or D(tau) implementation for more than 2 colors
  if (n_color != 2) {
    ALWAYS_EXPECTS((not has_jperp), "Error : has_jperp is true and we have {} colors instead of 2", n_color);
    ALWAYS_EXPECTS((not has_Dt), "Error : has_Dt is true and we have {} colors instead of 2", n_color);
  }

  // For numerical integration of the D0 and Jperp
  auto ramp = nda::zeros<double>(p.n_tau_k);
  for (auto n : range(p.n_tau_k)) { ramp(n) = n * beta / (p.n_tau_k - 1); }

  // Dynamical interactions
  if (has_Dt) {
    // Compute interaction kernels K(tau), K'(tau) by integrating D(tau)
    K      = gf<imtime>({beta, Boson, p.n_tau_k}, {n_color, n_color});
    Kprime = K;
    for (auto c1 : range(n_color)) {
      for (auto c2 : range(n_color)) {
        nda::array<dcomplex, 1> D_data = inputs.d0t.data()(range::all, c1, c2);
        auto first_integral            = nda::zeros<dcomplex>(p.n_tau_k);
        auto second_integral           = nda::zeros<dcomplex>(p.n_tau_k);
        // Trapezoidal integration
        for (int i = 1; i < D_data.size(); ++i) {
          first_integral(i)  = first_integral(i - 1) + (D_data(i) + D_data(i - 1)) / 2;
          second_integral(i) = second_integral(i - 1) + (first_integral(i) + first_integral(i - 1)) / 2;
        }
        // Noramlize by bin size
        first_integral *= beta / (p.n_tau_k - 1);
        second_integral *= (beta / (p.n_tau_k - 1)) * (beta / (p.n_tau_k - 1));
        // Enforce K(0) = K(beta) = 0
        Kprime.data()(range::all, c1, c2) = first_integral - second_integral(p.n_tau_k - 1) / beta;
        K.data()(range::all, c1, c2)      = second_integral - ramp * second_integral(p.n_tau_k - 1) / beta;
        // Renormalize U and mu
        if (c1 != c2) U(c1, c2) -= real(2 * Kprime.data()(0, c1, c2));
      }
      mu(c1) += real(Kprime.data()(0, c1, c1));
    }
  }

  // J_perp interactions
  if (has_jperp) {
    Jperp = inputs.jperpt;
    if (not has_Dt)
      rot_inv = false;
    else {
      Kprime_spin = gf<imtime>({beta, Boson, p.n_tau_k}, {n_color, n_color}); // used in computation of F(tau)
      // Integrate Jperp to obtain the S_z.S_z part of K'(tau) (called Kprime_spin)
      auto Kprime_J                  = Jperp;
      nda::array<dcomplex, 1> J_data = Jperp.data()(range::all, 0, 0);
      auto first_integral            = nda::zeros<dcomplex>(p.n_tau_k);
      // Trapezoidal integration
      for (int i = 1; i < J_data.size(); ++i) {
        first_integral(i) = first_integral(i - 1) + (J_data(i) + J_data(i - 1)) / 2;
      }
      // Noramlize by bin size
      first_integral *= beta / (p.n_tau_k - 1);
      // Enforce Kprime_J(beta/2) = 0
      Kprime_J.data()(range::all, 0, 0) = first_integral - first_integral((p.n_tau_k - 1) / 2);
      // Kprime_spin = +/- Kprime_J depending on color
      for (auto c1 : range(n_color)) {
        for (auto c2 : range(n_color)) {
          Kprime_spin.data()(range::all, c1, c2) = (c1 == c2 ? 1 : -1) * Kprime_J.data()(range::all, 0, 0) / 4;
        }
      }
      auto Kprime_0 = gf<imtime>({beta, Boson, p.n_tau_k}, {n_color, n_color});
      Kprime_0      = Kprime - Kprime_spin;
      // The "remainder" Kprime_0 must be color-independent for there to be rotational invariance
      if (max_element(abs(Kprime_0.data()(range::all, 0, 0) - Kprime_0.data()(range::all, 0, 1))) > 1.e-13)
        rot_inv = false;
    }
  }

  // Report
  if (c.rank() == 0) {
    spdlog::info("mu = {} \nU = {}", mu, U);
    spdlog::info("Dynamical interactions = {}, J_perp interactions = {}", has_Dt, has_jperp);
    if (p.measure_ft and !rot_inv)
      spdlog::info("WARNING: Cannot measure F(tau) because spin-spin interaction is not rotationally invariant.");
  }

  // ................  Determinants .....................
  // Is there a non-zero Delta(tau)?
  for (auto const &bl : range(inputs.delta.size())) {
    if (max_element(abs(inputs.delta[bl].data())) > 1.e-13) has_delta = true;
  }
  if (not has_delta) {
    ALWAYS_EXPECTS(has_jperp, "Error : both J_perp(tau) and Delta(tau) are 0: there is nothing to expand.");
    if (c.rank() == 0) { spdlog::info("Delta(tau) is 0, running only spin moves."); }
  }

  // FIXME: real??
  delta = map([](gf_const_view<imtime> d) { return real(d); }, inputs.delta);
  for (auto const &bl : range(delta.size())) {
    if (!is_gf_real(delta[bl], 1e-10)) {
      if (c.rank() == 0) {
        spdlog::info("WARNING: The Delta(tau) block number {} is not real in tau space", bl);
        spdlog::info("WARNING: max(Im[Delta(tau)]) = {}", max_element(abs(imag(delta[bl].data()))));
        spdlog::info("WARNING: Disregarding the imaginary component in the calculation.");
      }
    }
    // Construct the detmanip object for block bl
    dets.emplace_back(delta_block_adaptor{real(delta[bl])}, p.det_init_size);
    // Set parameters
    dets.back().set_singular_threshold(p.det_singular_threshold);
    dets.back().set_n_operations_before_check(p.det_n_operations_before_check);
    dets.back().set_precision_warning(p.det_precision_warning);
    dets.back().set_precision_error(p.det_precision_error);
  }
} // work_data constructor

// Additional sign of the trace (computed from dets).
double trace_sign(work_data_t const &wdata) {
  double sign      = 1.0;
  auto const &dets = wdata.dets;
  // For every block we compute the sign of the permutation that takes
  // [(c_dag c) (c_dag c) (c_dag c) ...] with the cdag and c in increasing time order
  // (the reference order of the det) to the completely time-and-color-ordered list
  //  of operators (the order that makes the trace positive)
  for (auto bl : range(dets.size())) {
    auto s              = long(dets[bl].size());
    auto n_colors_in_bl = wdata.gf_struct[bl].second;
    std::vector<int> number_c_before(n_colors_in_bl, 0);
    std::vector<int> number_cdag_before(n_colors_in_bl, 0);
    if (s != 0) {
      // We first compute the sign of the permutation that takes
      // [(c_dag c) (c_dag c) (c_dag c) ...] with the c and c_dag time-ordered to
      // [(c_dag c_dag ... cdag)(c c ... c)] with the c and c_dag time-ordered
      if ((s * (s - 1) / 2) % 2 == 1) { sign *= -1; }
      // We then compute the sign of the permutation that color-orders the c ...
      for (int n = 0; n < s; ++n) {
        auto c_color               = dets[bl].get_y(n).second;
        int n_higher_colors_before = 0;
        for (int k = c_color + 1; k < n_colors_in_bl; k++) { n_higher_colors_before += number_c_before[k]; }
        if (n_higher_colors_before % 2 == 1) sign *= -1;
        number_c_before[c_color]++;
      }
      // ... the sign of the permutation that color-orders the c_dag ...
      for (int n = 0; n < s; ++n) {
        auto cdag_color            = dets[bl].get_x(n).second;
        int n_higher_colors_before = 0;
        for (int k = cdag_color + 1; k < n_colors_in_bl; k++) n_higher_colors_before += number_cdag_before[k];
        if (n_higher_colors_before % 2 == 1) sign *= -1;
        number_cdag_before[cdag_color]++;
      }
      // ... and the sign of the permutation that assembles the operators by color.
      int nb_transp         = 0;
      auto nb_ops_per_color = number_c_before;
      for (int k = 0; k < n_colors_in_bl; ++k) {
        for (int l = k + 1; l < n_colors_in_bl; ++l) { nb_transp += nb_ops_per_color[k] * nb_ops_per_color[l]; }
      }
      if (nb_transp % 2 == 1) sign *= -1;
      // Finally, we compute the sign of the permutation that takes
      // [Color 0 : (c_dag c_dag ... cdag)(c c ... c) ... Color n: (c_dag c_dag ... cdag)(c c ... c)]
      // with the c and c_dag time-ordered within each color to the completely time-and-color-ordered
      // list of operators. In practice, we compute the sign of the time ordering for each color,
      // ignoring the other colors.
      for (int k = 0; k < n_colors_in_bl; ++k) {
        int cdag_to_jump = 0;
        int idx_c = s - 1, idx_cdag = s - 1;
        while (idx_c >= 0) {
          auto color_c = dets[bl].get_y(idx_c).second;
          auto time_c  = dets[bl].get_y(idx_c).first;
          if (color_c == k) {
            bool keep_going = true;
            while (keep_going and (idx_cdag >= 0)) {
              auto color_cdag = dets[bl].get_x(idx_cdag).second;
              auto time_cdag  = dets[bl].get_x(idx_cdag).first;
              keep_going      = (color_cdag != k) or (time_cdag > time_c);
              if (keep_going) {
                idx_cdag--;
                if (color_cdag == k) cdag_to_jump++;
              }
            } // loop over cdag
            if (cdag_to_jump % 2 == 1) sign *= -1;
          } // if color == k
          idx_c--;
        } // loop over c
      }   // loop over colors
    }     // if block not empty
  }       // loop over blocks
  return sign;
} // sign computation

// Functions for checking if a time is already in det.
bool c_in_det(tau_t const &tau, det_t const &D) {
  if (D.size() == 0) return false;
  auto det_c_time  = [&](long i) { return D.get_y(i).first; };
  long det_index_c = lower_bound(det_c_time, D.size(), tau);
  return (det_index_c >= D.size()) ? false : (det_c_time(det_index_c) == tau);
}

bool cdag_in_det(tau_t const &tau, det_t const &D) {
  if (D.size() == 0) return false;
  auto det_cdag_time  = [&](long i) { return D.get_x(i).first; };
  long det_index_cdag = lower_bound(det_cdag_time, D.size(), tau);
  return (det_index_cdag >= D.size()) ? false : (det_cdag_time(det_index_cdag) == tau);
}
