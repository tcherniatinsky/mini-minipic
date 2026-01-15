/* _____________________________________________________________________ */
//! \file Managers.cpp

//! \brief Management of the operators

/* _____________________________________________________________________ */

#include "Managers.hpp"
#include "Operators.hpp"

namespace managers {

void initialize(const Params &params, ElectroMagn &em,
                std::vector<Particles> &particles) {
  // Momentum correction (to respect the leap frog scheme)
  if (params.momentum_correction) {

    std::cout << " > Apply momentum correction "
              << "\n"
              << std::endl;

    operators::interpolate(em, particles);
    
    operators::push_momentum(particles, -0.5 * params.dt);
  }
}

void iterate(const Params &params, ElectroMagn &em,
             std::vector<Particles> &particles, int it) {
  if (params.current_projection || params.n_particles > 0) {

    DEBUG("  -> start reset current");

    em.reset_currents(minipic::device);

    DEBUG("  -> stop reset current");
  }

  // Interpolate from global field to particles
  DEBUG("  -> start interpolate ");

  operators::interpolate(em, particles);

  DEBUG("  -> stop interpolate");

  // Push all particles
  DEBUG("  -> start push ");

  operators::push(particles, params.dt);

  DEBUG("  -> stop push");

  // Do boundary conditions on global domain
  DEBUG("  -> Patch 0: start pushBC");

  operators::pushBC(params, particles);

  DEBUG("  -> stop pushBC");

#if defined(MINI_MINIPIC_DEBUG)
  // check particles
  for (std::size_t is = 0; is < particles.size(); ++is) {
    particles[is].check(inf_m[0], sup_m[0], inf_m[1], sup_m[1], inf_m[2],
                        sup_m[2]);
  }
#endif

  // Projection in local field
  if (params.current_projection) {

    // Projection directly in the global grid
    DEBUG("  ->  start projection");

    operators::project(params, em, particles);

    DEBUG("  ->  stop projection");
  }

  // __________________________________________________________________
  // Sum all species contribution in the local and global current grids

  if (params.current_projection || params.n_particles > 0) {

    // Perform the boundary conditions for current
    DEBUG("  -> start current BC")

    operators::currentBC(params, em);

    DEBUG("  -> stop current BC")

  } // end if current projection

  // __________________________________________________________________
  // Maxwell solver

  if (params.maxwell_solver) {

	//Fill a View with all necessary data for antenna profiles
	Kokkos::View<double *> antenna_positions_view("antenna positions view", params.antenna_profiles_m.size());
	Kokkos::View<double ***> antenna_profiles_view("antenna profiles view", params.antenna_profiles_m.size(), em.Jz_h_m.extent(1), em.Jz_h_m.extent(2));
	auto antenna_positions_view_h = Kokkos::create_mirror_view(antenna_positions_view);
	auto antenna_profiles_view_h = Kokkos::create_mirror_view(antenna_profiles_view);
	
	//Fill antenna profiles on host
	const double t = it * params.dt;
	const double yfs = 0.5 * params.Ly + params.inf_y;
	const double zfs = 0.5 * params.Lz + params.inf_z;
	const double dx = params.dx;
	const double inf_x = params.inf_x;
	const double J_dual_zx_m = em.J_dual_zx_m;
	//Policy to run code on OpenMP
	typedef Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>> host_mdrange_policy;
	//Policy to run code on Cuda
	typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> device_mdrange_policy;
	Kokkos::parallel_for(host_mdrange_policy({0, 0, 0}, {params.antenna_profiles_m.size(), em.Jz_h_m.extent(1), em.Jz_h_m.extent(2)}),
        [=](const int iantenna, const int iy, const int iz) {
          const double x = params.antenna_positions_m[iantenna];
          const double y = (iy - em.J_dual_zy_m * 0.5) * params.dy + params.inf_y - yfs;
		  const double z = (iz - em.J_dual_zz_m * 0.5) * params.dz + params.inf_z - zfs;
		  antenna_positions_view_h(iantenna) = x;
          antenna_profiles_view_h(iantenna, iy, iz) =  params.antenna_profiles_m[iantenna](y, z, t);
        }
    );
    Kokkos::fence();
	
	//Copy to device
	Kokkos::deep_copy(antenna_positions_view, antenna_positions_view_h);
	Kokkos::deep_copy(antenna_profiles_view, antenna_profiles_view_h);
    
    ElectroMagn::view_t J = em.Jz_m;
  
    
    Kokkos::parallel_for(device_mdrange_policy({0, 0, 0}, {params.antenna_profiles_m.size(), em.Jz_h_m.extent(1), em.Jz_h_m.extent(2)}),
        KOKKOS_LAMBDA(const int iantenna, const int iy, const int iz) {
		  const double x = antenna_positions_view(iantenna);
          const int ix = std::floor((x - inf_x - J_dual_zx_m * 0.5 * dx) / dx);
          J(ix, iy, iz) = antenna_profiles_view(iantenna, iy, iz);
        }
    );

    // Solve the Maxwell equation
    DEBUG("  -> start solve Maxwell")

    operators::solve_maxwell(params, em);

    DEBUG("  -> stop solve Maxwell")

    // Boundary conditions on EM fields
    DEBUG("  -> start solve BC")

    operators::solveBC(params, em);

    DEBUG("  -> end solve BC")

  } // end test params.maxwell_solver
}

} // namespace managers
