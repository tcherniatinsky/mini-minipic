#ifndef MY_KOKKOS_GRAPHS_HPP
#define MY_KOKKOS_GRAPHS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

struct MaxwellNodeUpdateEx
{
  MaxwellNodeUpdateEx(const double dt_, const double dt_over_dy_, const double dt_over_dz_, const ElectroMagn::view_t Ex_, const ElectroMagn::view_t By_, const ElectroMagn::view_t Bz_, const ElectroMagn::view_t Jx_) 
    : dt(dt_)
    , dt_over_dy(dt_over_dy_)
    , dt_over_dz(dt_over_dz_)
    , Ex(Ex_)
    , By(By_)
    , Bz(Bz_)
    , Jx(Jx_)
  {}
  
  KOKKOS_FUNCTION void operator() (const int ix, const int iy, const int iz) const
  {
    Ex(ix, iy, iz) += -dt * Jx(ix, iy + 1, iz + 1) +
      dt_over_dy * (Bz(ix, iy + 1, iz) - Bz(ix, iy, iz)) -
      dt_over_dz * (By(ix, iy, iz + 1) - By(ix, iy, iz));
  }
private:
  double dt;
  double dt_over_dy;
  double dt_over_dz;
  
  ElectroMagn::view_t Ex;
  
  ElectroMagn::view_t By;
  ElectroMagn::view_t Bz;
  
  ElectroMagn::view_t Jx;
};

struct MaxwellNodeUpdateEy
{
  MaxwellNodeUpdateEy(const double dt_, const double dt_over_dx_, const double dt_over_dz_, const ElectroMagn::view_t Ey_, const ElectroMagn::view_t Bx_, const ElectroMagn::view_t Bz_, const ElectroMagn::view_t Jy_) 
    : dt(dt_)
    , dt_over_dx(dt_over_dx_)
    , dt_over_dz(dt_over_dz_)
    , Ey(Ey_)
    , Bx(Bx_)
    , Bz(Bz_)
    , Jy(Jy_)
  {}
  
  KOKKOS_FUNCTION void operator() (const int ix, const int iy, const int iz) const
  {
    Ey(ix, iy, iz) += -dt * Jy(ix + 1, iy, iz + 1) -
        dt_over_dx * (Bz(ix + 1, iy, iz) - Bz(ix, iy, iz)) +
        dt_over_dz * (Bx(ix, iy, iz + 1) - Bx(ix, iy, iz));
  }
private:
  double dt;
  double dt_over_dx;
  double dt_over_dz;
  
  ElectroMagn::view_t Ey;
  
  ElectroMagn::view_t Bx;
  ElectroMagn::view_t Bz;
  
  ElectroMagn::view_t Jy;
};

struct MaxwellNodeUpdateEz
{
  MaxwellNodeUpdateEz(const double dt_, const double dt_over_dx_, const double dt_over_dz_, const ElectroMagn::view_t Ez_, const ElectroMagn::view_t Bx_, const ElectroMagn::view_t By_, const ElectroMagn::view_t Jz_) 
    : dt(dt_)
    , dt_over_dx(dt_over_dx_)
    , dt_over_dz(dt_over_dz_)
    , Ez(Ez_)
    , Bx(Bx_)
    , By(By_)
    , Jz(Jz_)
  {}
  
  KOKKOS_FUNCTION void operator() (const int ix, const int iy, const int iz) const
  {
    Ez(ix, iy, iz) += -dt * Jz(ix + 1, iy + 1, iz) +
      dt_over_dx * (By(ix + 1, iy, iz) - By(ix, iy, iz)) -
      dt_over_dx * (Bx(ix, iy + 1, iz) - Bx(ix, iy, iz));
  }
private:
  double dt;
  double dt_over_dx;
  double dt_over_dz;
  
  ElectroMagn::view_t Ez;
  
  ElectroMagn::view_t Bx;
  ElectroMagn::view_t By;
  
  ElectroMagn::view_t Jz;
};

struct MaxwellNodeUpdateBx
{
  MaxwellNodeUpdateBx(const double dt_, const double dt_over_dy_, const double dt_over_dz_, const ElectroMagn::view_t Bx_, const ElectroMagn::view_t Ey_, const ElectroMagn::view_t Ez_) 
    : dt(dt_)
    , dt_over_dy(dt_over_dy_)
    , dt_over_dz(dt_over_dz_)
    , Bx(Bx_)
    , Ey(Ey_)
    , Ez(Ez_)
  {}
  
  KOKKOS_FUNCTION void operator() (const int ix, const int iy, const int iz) const
  {
    Bx(ix, iy, iz) += -dt_over_dy * (Ez(ix, iy, iz) - Ez(ix, iy - 1, iz)) +
      dt_over_dz * (Ey(ix, iy, iz) - Ey(ix, iy, iz - 1));
  }
private:
  double dt;
  double dt_over_dy;
  double dt_over_dz;
  
  ElectroMagn::view_t Bx;
  
  ElectroMagn::view_t Ey;
  ElectroMagn::view_t Ez;
};

struct MaxwellNodeUpdateBy
{
  MaxwellNodeUpdateBy(const double dt_, const double dt_over_dx_, const double dt_over_dz_, const ElectroMagn::view_t By_, const ElectroMagn::view_t Ex_, const ElectroMagn::view_t Ez_) 
    : dt(dt_)
    , dt_over_dx(dt_over_dx_)
    , dt_over_dz(dt_over_dz_)
    , By(By_)
    , Ex(Ex_)
    , Ez(Ez_)
  {}
  
  KOKKOS_FUNCTION void operator() (const int ix, const int iy, const int iz) const
  {
    By(ix, iy, iz) += -dt_over_dz * (Ex(ix, iy, iz) - Ex(ix, iy, iz - 1)) +
      dt_over_dx * (Ez(ix, iy, iz) - Ez(ix - 1, iy, iz));
  }
private:
  double dt;
  double dt_over_dx;
  double dt_over_dz;
  
  ElectroMagn::view_t By;
  
  ElectroMagn::view_t Ex;
  ElectroMagn::view_t Ez;
};

struct MaxwellNodeUpdateBz
{
  MaxwellNodeUpdateBz(const double dt_, const double dt_over_dx_, const double dt_over_dy_, const ElectroMagn::view_t Bz_, const ElectroMagn::view_t Ex_, const ElectroMagn::view_t Ey_) 
    : dt(dt_)
    , dt_over_dx(dt_over_dx_)
    , dt_over_dy(dt_over_dy_)
    , Bz(Bz_)
    , Ex(Ex_)
    , Ey(Ey_)
  {}
  
  KOKKOS_FUNCTION void operator() (const int ix, const int iy, const int iz) const
  {
    Bz(ix, iy, iz) += -dt_over_dx * (Ey(ix, iy, iz) - Ey(ix - 1, iy, iz)) +
      dt_over_dy * (Ex(ix, iy, iz) - Ex(ix, iy - 1, iz));
  }
private:
  double dt;
  double dt_over_dx;
  double dt_over_dy;
  
  ElectroMagn::view_t Bz;
  
  ElectroMagn::view_t Ex;
  ElectroMagn::view_t Ey;
};

inline auto create_solve_maxwell_graph(const Params &params, ElectroMagn &em)
{
    const double dt = params.dt;
    const double dt_over_dx = params.dt * params.inv_dx;
    const double dt_over_dy = params.dt * params.inv_dy;
    const double dt_over_dz = params.dt * params.inv_dz;

    /////     Solve Maxwell Ampere (E)
    // Electric field Ex (d,p,p)

    ElectroMagn::view_t Ex = em.Ex_m; //ElectroMagn::hostview_t Ex = em.Ex_h_m;
    ElectroMagn::view_t Ey = em.Ey_m; //ElectroMagn::hostview_t Ey = em.Ey_h_m;
    ElectroMagn::view_t Ez = em.Ez_m; //ElectroMagn::hostview_t Ez = em.Ez_h_m;

    ElectroMagn::view_t Bx = em.Bx_m; //ElectroMagn::hostview_t Bx = em.Bx_h_m;
    ElectroMagn::view_t By = em.By_m; //ElectroMagn::hostview_t By = em.By_h_m;
    ElectroMagn::view_t Bz = em.Bz_m; //ElectroMagn::hostview_t Bz = em.Bz_h_m;

    ElectroMagn::view_t Jx = em.Jx_m;
    ElectroMagn::view_t Jy = em.Jy_m;
    ElectroMagn::view_t Jz = em.Jz_m;

    const int nx_d = em.nx_d_m;
    const int ny_d = em.ny_d_m;
    const int nz_d = em.nz_d_m;
    const int nx_p = em.nx_p_m;
    const int ny_p = em.ny_p_m;
    const int nz_p = em.nz_p_m;
    
    MaxwellNodeUpdateEx maxwellNodeUpdateEx(dt, dt_over_dy, dt_over_dz, Ex, By, Bz, Jx);
    MaxwellNodeUpdateEy maxwellNodeUpdateEy(dt, dt_over_dx, dt_over_dz, Ey, Bx, Bz, Jy);
    MaxwellNodeUpdateEz maxwellNodeUpdateEz(dt, dt_over_dx, dt_over_dy, Ez, Bx, By, Jz);
    
    MaxwellNodeUpdateBx maxwellNodeUpdateBx(dt, dt_over_dy, dt_over_dz, Bx, Ey, Ez);
    MaxwellNodeUpdateBy maxwellNodeUpdateBy(dt, dt_over_dx, dt_over_dz, By, Ex, Ez);
    MaxwellNodeUpdateBz maxwellNodeUpdateBz(dt, dt_over_dx, dt_over_dy, Bz, Ex, Ey);

    typedef Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>> mdrange_policy3;

    auto graph = Kokkos::Experimental::create_graph([&](auto root) 
    {
        // update E
        auto node_update_Ex = root.then_parallel_for("update Ex in solve_maxwell", mdrange_policy3({0, 0, 0}, {nx_d, ny_p, nz_p}), maxwellNodeUpdateEx);
        auto node_update_Ey = root.then_parallel_for("update Ey in solve_maxwell", mdrange_policy3({0, 0, 0}, {nx_p, ny_d, nz_p}), maxwellNodeUpdateEy);
        auto node_update_Ez = root.then_parallel_for("update Ez in solve_maxwell", mdrange_policy3({0, 0, 0}, {nx_p, ny_p, nz_d}), maxwellNodeUpdateEz);
        // update B
        Kokkos::Experimental::when_all(node_update_Ey, node_update_Ez).then_parallel_for("update Bx in solve_maxwell", mdrange_policy3({0, 1, 1}, {nx_p, ny_d, nz_d}), maxwellNodeUpdateBx);
        Kokkos::Experimental::when_all(node_update_Ex, node_update_Ez).then_parallel_for("update By in solve_maxwell", mdrange_policy3({1, 0, 1}, {nx_d, ny_p, nz_d}), maxwellNodeUpdateBy);
        Kokkos::Experimental::when_all(node_update_Ey, node_update_Ex).then_parallel_for("update Bz in solve_maxwell", mdrange_policy3({1, 1, 0}, {nx_d, ny_d, nz_p}), maxwellNodeUpdateBz);
    });
    graph.instantiate();
    
    return graph;
}

#endif // MY_KOKKOS_GRAPHS_HPP
