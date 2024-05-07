using POMDPs
using POMDPTools
using QuickPOMDPs: QuickPOMDP
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using LinearAlgebra
using QMDP
using StaticArrays

global victim_state = SA[7,6]
global N = 10


function calcTransieverReading(s::SVector{2, Int64})
    ## Flux distance
    d = norm(victim_state - s)
    dx = s[1] - victim_state[1]
    dy = s[2] - victim_state[2]
    if dx == 0
        arclength = abs(dy)
    else
        arclength = (pi - 2*acos(abs(dx)/d)) * d^2 / (2*abs(dx))
    end

    ## Flux direction
    dir_angle = 0
    if dx == 0
        dir_angle = 90
    else
        dir_angle = rad2deg(abs(2*acos(abs(dx)/d) - pi/2))
    end
    dir = SA[1,0]
    
    # Quadrant adjustment
    r = 0
    if dx != 0
        r = d^2/(2*abs(dx))
        #Left of victim
        if dx < 0
            # Lower left and Upper right
            if (dy < 0 && abs(dx) >= r) || (dy >= 0 && abs(dx) < r)
                if dir_angle <= 30
                    dir = SA[1,0]
                elseif 30 < dir_angle < 60
                    dir = SA[1,-1]
                elseif dir_angle >= 60
                    dir = SA[0,-1]
                end
            #Lower right and Upper left
            elseif (dy < 0 && abs(dx) < r) || (dy >= 0 && abs(dx) >= r)
                if dir_angle <= 30
                    dir = SA[1,0]
                elseif 30 < dir_angle < 60
                    dir = SA[1,1]
                elseif dir_angle >= 60
                    dir = SA[0,1]
                end
            end
        elseif dx > 0
            # Lower right and Upper left
            if (dy < 0 && abs(dx) >= r) || (dy > 0 && abs(dx) < r)
                if dir_angle <= 30
                    dir = SA[-1,0]
                elseif 30 < dir_angle < 60
                    dir = SA[-1,-1]
                elseif dir_angle >= 60
                    dir = SA[0,-1]
                end
            #Lower left and Upper right
            elseif (dy < 0 && abs(dx) < r) || (dy > 0 && abs(dx) >= r)
                if dir_angle <= 30
                    dir = SA[-1,0]
                elseif 30 < dir_angle < 60
                    dir = SA[-1,1]
                elseif dir_angle >= 60
                    dir = SA[0,1]
                end
            end

        end
    else
        if dy > 0
            dir = SA[0,-1]
        elseif dy < 0
            dir = SA[0,1]
        end
    end

    return (arclength, dir)
end

function rescue_observations()
    os = []
     
    for s in [SA[x, y] for x in 1:N, y in 1:N]
        reading = calcTransieverReading(s)
        push!(os,reading)
    end

    return unique(push!(os,(-1, SA[1,0])))
end

function trans(s::SVector, a::SVector)
    adj_states = []
    probs = []
    allow_desired_a = false
    #no_transition = [SA[3,6],SA[4,6],SA[5,6],SA[6,6],SA[7,6],SA[8,6],SA[9,6],SA[5,3],SA[6,3],SA[7,3],SA[8,3],SA[9,3]]
    no_transition = [SA[3,3],SA[4,3],SA[5,3],SA[6,3],SA[7,3],SA[6,9],SA[7,9],SA[8,9],SA[9,9]]
    for action in actions(rescue)[1:8]
        sp = s + action
        if all(x -> 0 < x < N+1, sp) && !(sp in no_transition) 
            push!(adj_states, sp)
        end
    end

    if s+a in adj_states
        allow_desired_a = true
    else
        allow_desired_a = false
    end


    for adj_state in adj_states
        if allow_desired_a == true
            if adj_state == s+a
                push!(probs,0.7)
            else
                push!(probs, 0.3/(length(adj_states)-1))
            end
        else
            push!(probs, 1/length(adj_states))
        end
    end
    return SparseCat(adj_states, probs)
end

function trans(s::SVector, a::Symbol)
    if s == victim_state && a == :dig
        return Deterministic(:found)
    else
        return Deterministic(s)
    end
end

function trans(s::Symbol, a::SVector)
    return Deterministic(s)
end

function trans(s::Symbol, a::Symbol)
    return Deterministic(s)
end



###################
##### Rescue ######
###################

rescue = QuickPOMDP(

states = union([SA[x, y] for x in 1:N, y in 1:N],[:found]),

actions = (SA[1,0], SA[-1,0], SA[0,1], SA[0,-1], SA[1,1], SA[1,-1], SA[-1,1], SA[-1,-1], :look, :dig),

observations = rescue_observations(),



# transition should be a function that takes in s and a and returns the distribution of s'
transition = trans,

# observation should be a function that takes in s, a, and sp, and returns the distribution of o
observation = function (a, sp)
    if sp == :found
        reading = (-1,SA[1,0])
    else
        reading = calcTransieverReading(sp)
    end
    
    n_os = length(observations(m))
    if a == :look 
        return SparseCat(SVector{n_os}(push!(filter(o -> o != reading, observations(m)), reading)), SVector{n_os}(append!(0.1*ones(Float64, n_os-1)/(n_os-1), 0.9)))
    else
        return SparseCat(SVector{n_os}(push!(filter(o -> o != reading, observations(m)), reading)),  SVector{n_os}(append!(0.8*ones(Float64, n_os-1)/(n_os-1), 0.2)))
    end 

end,

reward = function (s, a)
    if a == :look
        return -2
    elseif a == :dig && s == victim_state
        return 100
    elseif a == :dig
        return -10
    else
        return -1
    end
end,

initialstate =  Deterministic(SA[1,1]),

discount = 0.90,
isterminal = s -> s == :found,
)

m = rescue






