using POMDPs
using POMDPTools
using QuickPOMDPs: QuickPOMDP
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using LinearAlgebra
using QMDP
using DiscreteValueIteration
using CSV
using DataFrames

# include("HW62.jl")
include("rescuePOMDP.jl")

m = rescue


struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function update(up::HW6Updater, b::DiscreteBelief, a, o)
    pomdp = up.pomdp
    state_space = b.state_list
    bp = zeros(length(state_space))
    for (si, s) in enumerate(state_space)
        if pdf(b, s) > 0.0
            td = transition(pomdp, s, a)

            for (sp, tp) in weighted_iterator(td)
                spi = stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o) # shortcut for observation probability from POMDPModelTools
                bp[spi] += op * tp * b.b[si]
            end
        end
    end

    bp_sum = sum(bp)

    if bp_sum == 0.0
        error("""
              Failed discrete belief update: new probabilities sum to zero.

              b = $b
              a = $a
              o = $o

              Failed discrete belief update: new probabilities sum to zero.
              """)
    end

    # Normalize
    bp ./= bp_sum

    return DiscreteBelief(pomdp, b.state_list, bp)
end



# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.



# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end


#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    bvec = beliefvec(b)
    idx = argmax([dot(alpha,bvec) for alpha in p.alphas])

    return p.alpha_actions[idx]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------


function qmdp_solve(m, discount=discount(m))

    # Fill in Value Iteration to compute the Q-values
    
    acts = actiontype(m)[]
    alphas = Vector{Float64}[]
    
    # V = zeros(length(states(m))) # this would be a good container to use for your value function
    Vprime =  zeros(length(states(m)))
    delta = 1
    epsilon = 1e-3
    while delta > epsilon
        delta = 0
        
        
        for s in ordered_states(m)
            V = Vprime[stateindex(m,s)]
            Qmax = -1e6
            for a in ordered_actions(m)
                
                    # o = observation(m,a,sp)   T ,sp
                     Q = reward(m,s,a) + discount*sum(T(m,s,a,sp)*(Vprime[j]) for (j,sp) in enumerate(states(m)))
                    Qmax = max(Qmax,Q)
            end
            Vprime[stateindex(m,s)] = Qmax   
            delta = max(delta, abs(Qmax-V))
            
         end
    end
    # @show Vprime
    
    for a in ordered_actions(m)
        alpha = zeros(length(states(m)))
        for s in ordered_states(m)
             
                alpha[stateindex(m,s)] = reward(m,s,a) + discount*sum(T(m,s,a,sp)*(Vprime[j]) for(j,sp) in enumerate(states(m)))
            
        end
        push!(acts,a)
        push!(alphas,alpha)
        
    end

            # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
        
    
    return HW6AlphaVectorPolicy(alphas, acts)
end




up = DiscreteUpdater(m)




qmdp_p = qmdp_solve(m)
@show mean(simulate(RolloutSimulator(), m, qmdp_p, up) for _ in 1:1)     # Should be approximately 66 =#



hr = HistoryRecorder(max_steps=50)
using BasicPOMCP
function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
    solver = POMCPSolver(tree_queries=1000,#20
                         c=30,#50.0,
                        #  max_time = 0.2, 
                         default_action= rand(actions(m)),
                         estimate_value= FOValue(ValueIterationSolver()))  #FORollout())  #FunctionPolicy(s->rand(actions(m)))
    return solve(solver, m)
end
pomcp_p = pomcp_solve(m)
@show mean(simulate(RolloutSimulator(), m, pomcp_p, up) for _ in 1:25)

heuristic = FunctionPolicy(function (b)
    
bels=zeros(length(states(m)))
i=1
for s in states(m)
    bels[i] = pdf(b,s)
    i = i+1
end
state = states(m)[argmax(bels)]

if state != :found
    read = calcTransieverReading(state)

    if pdf(b,victim_state)>0.06
        return :dig
    else 
        return read[2] 
        
    end
else
    return rand(actions(m))
end
end                        
)

h = simulate(hr, m, heuristic, up)

step_states = []
step_actions = []
for i = 1:length(h)
    st = h[i]
    state = st[:s]
    action = st[:a]
    push!(step_states,state)
    push!(step_actions,action)
end
display(step_states)
display(step_actions)
accumulated_reward = discounted_reward(h)

# Create a DataFrame with step states and step actions
df = DataFrame(State = step_states, Action = step_actions)
reward_df = DataFrame(Title = "Accumulated Reward", Reward = accumulated_reward)

# Write the DataFrame to a CSV file
CSV.write("step_data.csv", df)
CSV.write("reward.csv", reward_df)

sarsop_p = solve(SARSOPSolver(), m)
@show mean(simulate(RolloutSimulator(), m, sarsop_p, up) for _ in 1:3)

