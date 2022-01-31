include("../HexWorld/DiscreteMDP.jl")
include("../HexWorld/HexWorld.jl")
include("../helpers/SimpleGame/SimpleGame.jl")
include("Visualize.jl")

using Random
using JuMP
using LinearAlgebra
using GridInterpolations
using DataFrames
using IndexedTables

# structure of Markov Game
struct MG
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    T  # transition function
    R  # joint reward function
end

struct PredatorPreyHexWorldMG
    hexes::Vector{Tuple{Int,Int}}   # include coordinates of all hex cells
    hexWorldDiscreteMDP::DiscreteMDP
end

n_agents(mg::PredatorPreyHexWorldMG) = 2 # game include 2 agents: predator - prey

# state in order, from 1 to length(mg.hexes), of agent i
ordered_states(mg::PredatorPreyHexWorldMG, i::Int) = vec(collect(1:length(mg.hexes)))

# chỉnh hợp tập state của các agents 
ordered_states(mg::PredatorPreyHexWorldMG) = vec(collect(Iterators.product([ordered_states(mg, i) for i in 1:n_agents(mg)]...)))

# action in order, from 1 to n_actions, of agent i
ordered_actions(mg::PredatorPreyHexWorldMG, i::Int) = vec(collect(1:n_actions(mg.hexWorldDiscreteMDP)))

# chỉnh hợp tập action của các agents 
ordered_joint_actions(mg::PredatorPreyHexWorldMG) = vec(collect(Iterators.product([ordered_actions(mg, i) for i in 1:n_agents(mg)]...)))

# number of actions
n_actions(mg::PredatorPreyHexWorldMG, i::Int) = length(ordered_actions(mg, i))

# number of joint actions
n_joint_actions(mg::PredatorPreyHexWorldMG) = length(ordered_joint_actions(mg))


# calculate probability of transition from s to s' with action a of 2 agents
function transition(mg::PredatorPreyHexWorldMG, s, a, s′)

    # when the prey is captured, it will be transposed to a random hex cell in hex map with probability = 1/12
    if s[1] == s[2]
        prob = Float64(s′[1] == s[1]) / length(mg.hexes)
    else
        prob = mg.hexWorldDiscreteMDP.T[s[1], a[1], s′[1]] * mg.hexWorldDiscreteMDP.T[s[2], a[2], s′[2]]
    end

    return prob
end

# calculate reward for agent i in state s
function reward(mg::PredatorPreyHexWorldMG, i::Int, s, a)
    r = 0.0

    if i == 1
        # Predator get -1 for moving and 10 for catching the prey.
        if s[1] == s[2]
            return 10.0
        else
            return -1.0
        end
    elseif i == 2
        # Prey get -1 for moving and -100 for being caught.
        if s[1] == s[2]
            r = -100.0
        else
            r = -1.0
        end
    end

    return r
end

# calculate the rewards of 2 agents in state s
function joint_reward(mg::PredatorPreyHexWorldMG, s, a)
    return [reward(mg, i, s, a) for i in 1:n_agents(mg)]
end


# initialize MG from PredatorPreyHexWorldMG
function MG(mg::PredatorPreyHexWorldMG)
    return MG(
        mg.hexWorldDiscreteMDP.γ,
        vec(collect(1:n_agents(mg))),
        ordered_states(mg),
        [ordered_actions(mg, i) for i in 1:n_agents(mg)],
        (s, a, s′) -> transition(mg, s, a, s′),
        (s, a) -> joint_reward(mg, s, a)
    )
end

# initialize PredatorPreyHexWorldMG from HexWorldMDP
function PredatorPreyHexWorldMG(hexes::Vector{Tuple{Int,Int}},
    r_bump_border::Float64,
    p_intended::Float64,
    γ::Float64)
    hexWorld = HexWorldMDP(hexes,
        r_bump_border,
        p_intended,
        Dict{Tuple{Int64,Int64},Float64}(),
        γ)
    mdp = hexWorld.mdp
    return PredatorPreyHexWorldMG(hexes, mdp)
end

# structure for visualization the result
struct VisualizePPHW
    model # calculate the probability of each action over each iteration
    policy # agents policy per iteration
    states # agents states per iteration
    rewards # agents rewards per iteration
    captured # save the iteration that the prey is captured
    function VisualizePPHW(k_max)
        k_max += 1
        model = [DataFrame(east = zeros(k_max), north_east = zeros(k_max), north_west = zeros(k_max), west = zeros(k_max), south_west = zeros(k_max), south_east = zeros(k_max)),
            DataFrame(east = zeros(k_max), north_east = zeros(k_max), north_west = zeros(k_max), west = zeros(k_max), south_west = zeros(k_max), south_east = zeros(k_max))]
        policy = [DataFrame(east = zeros(k_max), north_east = zeros(k_max), north_west = zeros(k_max), west = zeros(k_max), south_west = zeros(k_max), south_east = zeros(k_max)),
            DataFrame(east = zeros(k_max), north_east = zeros(k_max), north_west = zeros(k_max), west = zeros(k_max), south_west = zeros(k_max), south_east = zeros(k_max))]
        states = Vector{Tuple{Int64,Int64}}()
        rewards = Vector{Tuple{Int64,Int64}}()
        captured = Vector{Int}()

        # initialize for iteration 0
        push!(rewards, (0, 0))
        model[1][1, :] .= [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
        model[2][1, :] .= [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]

        return new(model, policy, states, rewards, captured)
    end
end

# const HexWorldRBumpBorder = -1.0 # Reward for falling off hex map
# const HexWorldPIntended = 0.7 # Probability of going intended direction
# const HexWorldDiscountFactor = 0.9

function PredatorPreyHexWorld()
    PredatorPreyHexWorld = PredatorPreyHexWorldMG(
        [
            (-1, 2), (0, 2), (1, 2),
            (-1, 1), (1, 1), (3, 1), (4, 1),
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        ],
        HexWorldRBumpBorder,    # we don't use this
        HexWorldPIntended,
        HexWorldDiscountFactor
    )
    return PredatorPreyHexWorld
end


struct MGPolicy
    p # dictionary mapping states to simple game policies
    MGPolicy(p::Base.Generator) = new(Dict(p))
end

# return probability agent i will do action ai in state s
(πi::MGPolicy)(s, ai) = πi.p[s](ai)

# return probability agent i will do action ai
(πi::SimpleGamePolicy)(s, ai) = πi(ai)

# π: MGPolicy
# return probability 2 agents will do action a in state s
probability(𝒫::MG, s, π, a) = prod(πj(s, aj) for (πj, aj) in zip(π, a))

# evaluate reward of agent i in state s = reward (agent i in state s)* probability ??
reward(𝒫::MG, s, π, i) =
    sum(𝒫.R(s, a)[i] * probability(𝒫, s, π, a) for a in joint(𝒫.𝒜))

# sum of probabilities that 2 agents can switch from s to s'
transition(𝒫::MG, s, π, s′) =
    sum(𝒫.T(s, a, s′) * probability(𝒫, s, π, a) for a in joint(𝒫.𝒜))



mutable struct MGFictitiousPlay
    𝒫 # Markov game
    i # agent index
    Qi # state-action value estimates
    Ni # state-action counts
end

function MGFictitiousPlay(𝒫::MG, i)
    ℐ, 𝒮, 𝒜, R = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R

    # Qi = [(s, a) => reward of agent i]
    Qi = Dict((s, a) => R(s, a)[i] for s in 𝒮 for a in joint(𝒜))

    # initialize with all state-action counts = 1
    # Ni = [(agent j, state s, action aj) => 1]
    Ni = Dict((j, s, aj) => 1.0 for j in ℐ for s in 𝒮 for aj in 𝒜[j])
    return MGFictitiousPlay(𝒫, i, Qi, Ni)
end


function (πi::MGFictitiousPlay)(s)
    𝒫, i, Qi = πi.𝒫, πi.i, πi.Qi

    # 𝒫: Markov game
    ℐ, 𝒮, 𝒜, T, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ

    # count the number of times each action has been performed at state s
    # return SimpleGamePolicy(Dict( action ai => Ni[i, s, ai]) )
    πi′(i, s) = SimpleGamePolicy(ai => πi.Ni[i, s, ai] for ai in 𝒜[i])

    # MGPolicy(Dict{s => SimpleGamePolicy}
    πi′(i) = MGPolicy(s => πi′(i, s) for s in 𝒮)

    π = [πi′(i) for i in ℐ]

    # display(π[1].p[(1,2)](4))

    # estimate the utility of agent i: state value estimates for s
    U(s, π) = sum(πi.Qi[s, a] * probability(𝒫, s, π, a) for a in joint(𝒜))

    # estimate the Q-value of agent i: reward of state s + state value estimates for the next state (s' - all possible states)
    Q(s, π) = reward(𝒫, s, π, i) + γ * sum(transition(𝒫, s, π, s′) * U(s′, π)
                                           for s′ in 𝒮)


    # joint : replace π[i] with πi
    # joint return [SimpleGamePolicy, MGPolicy ] or [MGPolicy, SimpleGamePolicy]
    # SimpleGamePolicy(Dict(ai => 1.0)), all actions different ai => 0

    # estimate the Q-value of agent i if it chooses action ai for the next turn (ai => 1.0)
    Q(ai) = Q(s, joint(π, SimpleGamePolicy(ai), i))

    # return action that has the highest Q-value 
    ai = argmax(Q, 𝒫.𝒜[πi.i])

    # return the policy with probability of action ai => 1.0 
    return SimpleGamePolicy(ai)
end

# update policy πi after transform from s to s' with action a
function update!(πi::MGFictitiousPlay, s, a, s′, v, iteration)
    𝒫, i, Qi = πi.𝒫, πi.i, πi.Qi
    ℐ, 𝒮, 𝒜, T, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.T, 𝒫.R, 𝒫.γ

    # update Ni
    for (j, aj) in enumerate(a)
        πi.Ni[j, s, aj] += 1
    end

    # update visualize
    v.policy[i][iteration, a[i]] = 1

    totalCount = sum(πi.Ni[i, S, ai] for S in 𝒮 for ai in 𝒜[i])
    for ai in 𝒜[i]
        v.model[i][iteration+1, ai] = sum(πi.Ni[i, S, ai] for S in 𝒮) / totalCount
    end

    # count the number of times each action has been performed at state s
    # return SimpleGamePolicy(Dict( action ai => Ni[i, s, ai]) )
    πi′(i, s) = SimpleGamePolicy(ai => πi.Ni[i, s, ai] for ai in 𝒜[i])

    # MGPolicy(Dict{s => SimpleGamePolicy}
    πi′(i) = MGPolicy(s => πi′(i, s) for s in 𝒮)

    π = [πi′(i) for i in ℐ]

    # estimate the utility of agent i: state value estimates for s
    U(π, s) = sum(πi.Qi[s, a] * probability(𝒫, s, π, a) for a in joint(𝒜))

    # calculate the state-action value estimates of agent i: 
    # reward of state s + state value estimates for the next state s' when take action a
    Q(s, a) = R(s, a)[i] + γ * sum(T(s, a, s′) * U(π, s′) for s′ in 𝒮)

    # update Qi
    for a in joint(𝒜)
        πi.Qi[s, a] = Q(s, a)
    end
end


# random the next turn
function randstep(𝒫::MG, s, a)
    # random s' base on the distribution of the transition 
    s′ = rand(SetCategorical(𝒫.𝒮, [𝒫.T(s, a, s′) for s′ in 𝒫.𝒮]))

    # reward in state s
    r = 𝒫.R(s, a)

    return s′, r
end


# random initial state of 2 agents
function randInitialState(𝒮)
    s = rand(𝒮)

    while s[1] == s[2]
        s = rand(𝒮)
    end
    return s
end

# simulate the joint policy π for k_max steps
function simulate(𝒫::MG, π, k_max)

    v = VisualizePPHW(k_max)

    s = randInitialState(𝒫.𝒮)

    push!(v.states, s)
    for k = 1:k_max

        # choose action a for the next turn
        a = Tuple(πi(s)() for πi in π)

        # random state 
        s′, r = randstep(𝒫, s, a)
        # println(s," => ",s′)
        for πi in π
            # update lại policy
            update!(πi, s, a, s′, v, k)
        end

        # update reward visualize

        # update reward = 0 if the agent does not move
        if (s[1] != s[2])
            if (s[1] == s′[1])
                r[1] = 0
            end
            if (s[2] == s′[2])
                r[2] = 0
            end
        else # the prey is captured
            push!(v.captured, k)
        end
        if (k > 1)
            r[1] += v.rewards[k][1]
            r[2] += v.rewards[k][2]
        end
        push!(v.rewards, Tuple(r))
        push!(v.states, s′)

        # use s′ as current state for next iteration 
        s = s′

    end

    # just for visualization purposes
    a = Tuple(πi(s)() for πi in π)
    for i in 1:2
        v.policy[i][k_max+1, a[i]] = 1
    end

    return v, π
end


#-------------- main ---------------

p = PredatorPreyHexWorld()
mg = MG(p)
π = [MGFictitiousPlay(mg, i) for i in 1:2]
k_max = 10

v, policy = simulate(mg, π, k_max)


# choose 1 for 2 visualizations (uncomment from the comments starting with @@@)



# @@@ use to visualize step by step (limit k_max <= 10)
drawStepbyStepPredatorPreyHW(v.states, v.rewards, v.captured, k_max)


# @@@ used for general visualization
# visualizeGeneralPredatorPreyHW(v)

