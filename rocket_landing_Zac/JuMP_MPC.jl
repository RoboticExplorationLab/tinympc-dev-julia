mutable struct SolnStats
    max_dyn_violation::Float64
    avg_dyn_violation::Float64
    max_bnd_violation::Float64
    avg_bnd_violation::Float64
    max_soc_violation::Float64  
    avg_soc_violation::Float64
    tracking_error::Float64
end

function stage_cost(p::NamedTuple, x, u, k)
dx = x - p.Xref[:,k]
du = u - p.Uref[:,k]
return 0.5 * dx' * p.Q * dx + 0.5 * du' * p.R * du
end

function term_cost(p::NamedTuple, x)
dx = x - p.Xref[:,p.Nh]
return 0.5 * dx' * p.Qf * dx
end

function stage_cost_expansion(p::NamedTuple, k)
dx = -p.Xref[:,k]
du = -p.Uref[:,k]
return p.Q, p.Q * dx, p.R, p.R * du  # Hessian and gradient
end

function term_cost_expansion(p::NamedTuple)
dx = -p.Xref[:,p.Nh]
return p.Qf, p.Qf * dx
end

function solve_mpc_JuMP(optimizer, params, X, U; warm_start=true)
  Nh = params.Nh
  nx = params.nx
  nu = params.nu
  A = params.Adyn
  B = params.Bdyn
  f = params.fdyn
  α_max = params.c_cone[1]  # Thrust angle constraint
  NN = Nh*nx + (Nh-1)*nu  # number of decision variables
  x0_ = 1*X[:,1]  # initial state
  
  # we compress x and u into one single decision variable z, 
  # then these are their indices
  inds = reshape(1:(nx+nu)*Nh,nx+nu,Nh)  
  xinds = [z[1:nx] for z in eachcol(inds)]
  uinds = [z[nx+1:end] for z in eachcol(inds)][1:Nh-1]    
  
  model = Model(optimizer)
  
  @variable(model, z[1:NN])  # z is all decision variables (X U)
  if warm_start
      z_ws = zeros(NN,1)
      for j = 1:Nh-1
          z_ws[xinds[j]] .= X[:,j]
          z_ws[uinds[j]] .= U[:,j]
      end
      z_ws[xinds[Nh]] .= X[:,Nh]
      set_start_value.(z, z_ws)
  end
  
  P = zeros(NN, NN)
  q = zeros(NN, 1) 
  # Cost function   
  for j = 1:Nh-1
      P[(j-1)*(nx+nu).+(1:nx),(j-1)*(nx+nu).+(1:nx)], q[(j-1)*(nx+nu).+(1:nx)], 
      P[(j-1)*(nx+nu)+nx.+(1:nu),(j-1)*(nx+nu)+nx.+(1:nu)], q[(j-1)*(nx+nu)+nx.+(1:nu)] = stage_cost_expansion(params, j)
  end    
  P[end-nx+1:end,end-nx+1:end], q[end-nx+1:end] = term_cost_expansion(params)
  @objective(model, Min, 0.5*dot(z,P,z) + dot(q,z))

  # Dynamics Constraints
  for k = 1:Nh-1
      @constraint(model, A*z[xinds[k]] .+ B*z[uinds[k]] .+ f .== z[xinds[k+1]])
  end
  
  # Initial condition 
  @constraint(model, z[xinds[1]] .== x0_)
  
  # Thrust angle constraint (SOC): norm([u1,u2]) <= α_max * u3
  if params.ncu_cone > 0 
    for k = 1:Nh-1
        u1,u2,u3 = z[uinds[k]]
        @constraint(model, [α_max * u3, u1, u2] in JuMP.SecondOrderCone())
    end
  end
  
  # Thrust angle constraint (SOC): norm([u1,u2]) <= α_max * u3
  if params.ncx_cone > 0 
    for k = 1:Nh
        x1,x2,x3,x4,x5,x6 = z[xinds[k]]
        @constraint(model, [0.8 * x3, x1, x2] in JuMP.SecondOrderCone())
    end
  end

  # State Constraints
  if params.ncx > 0 
    for k = 1:Nh
      @constraint(model, z[xinds[k]] .<= params.x_max)
      @constraint(model, z[xinds[k]] .>= params.x_min)
    end  
  end

  # Input Constraints
  if params.ncu > 0 
    for k = 1:Nh-1
      @constraint(model, z[uinds[k]] .<= params.u_max)
      @constraint(model, z[uinds[k]] .>= params.u_min)
    end  
  end

  # Goal constraint
  if params.ncg > 0 
      @constraint(model, z[xinds[N]] .== zeros(nx))
  end    

  optimize!(model)   
  # termination_status(model) == INFEASIBLE && print("Other solver says INFEASIBLE\n")

  # get results
  for j = 1:Nh-1
      X[:,j] .= value.(z[xinds[j]]) 
      U[:,j] .= value.(z[uinds[j]]) 
  end    
  X[:,Nh] .= value.(z[xinds[Nh]])
  # display(MOI.get(model, MOI.SolveTimeSec()))
  return U[:,1]
end

function mat_from_vec(X::Vector{Vector{Float64}})::Matrix
    # convert a vector of vectors to a matrix 
    Xm = hcat(X...)
    return Xm 
end

# function to check if the solution satisfies the dynamics and constraints
function check_solution!(info::SolnStats, opti, params, X, U)
    # check if X and U satisfy the dynamics
    for k = 1:params.Nh-1
        temp = norm(params.Adyn*X[:,k] + params.Bdyn*U[:,k] + params.fdyn - X[:,k+1])
        if temp > info.max_dyn_violation
            info.max_dyn_violation = temp
        end
        info.avg_dyn_violation += temp/(params.Nh-1)/(NTOTAL-1)
    end
    # check if U satisfies the bounds
    for k = 1:params.Nh-1
        temp = norm(max.(abs.(U[:,k]) - params.u_max, 0))
        if temp > info.max_bnd_violation
            info.max_bnd_violation = temp
        end
        info.avg_bnd_violation += temp/(params.Nh-1)/(NTOTAL-1)
    end
    # check if U satisfies the thrust angle constraint
    for k = 1:params.Nh-1
        temp = max.(norm([U[1,k],U[2,k]]) - params.c_cone[1]*U[3,k], 0)
        if temp > info.max_soc_violation
            info.max_soc_violation = temp
        end
        info.avg_soc_violation += temp/(params.Nh-1)/(NTOTAL-1)
    end
end

