# import Pkg
# Pkg.add("Parameters")

using Random
using Parameters: @with_kw
using JuMP
using Ipopt

# Pkg.add("Gadfly")
# Pkg.add("DataFrames")
# using DataFrames
# using Gadfly
# import Base.rand

# plot(data::DataFrames, mapping::Dict(), Element::element, Geometrics())
# plot(x=Base.rand(20), y=Base.rand(20), Geom.point)

struct SimpleGame
    γ # discount factor
    ℐ # agents
    𝒜 # joint action space
    R # joint reward function
end

struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k, v) in zip(keys(p), vs)))
    end

    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

struct POMG
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    𝒪  # joint observation space
    T  # transition function
    O  # joint observation function
    R  # joint reward function
end

struct BoolDistribution
    p::Float64 # probability of true
end

pdf(d::BoolDistribution, s::Bool) = s ? d.p : 1.0 - d.p
rand(rng::AbstractRNG, d::BoolDistribution) = rand(rng) <= d.p
iterator(d::BoolDistribution) = [true, false]
Base.:(==)(d1::BoolDistribution, d2::BoolDistribution) = d1.p == d2.p
Base.hash(d::BoolDistribution, u::UInt64 = UInt64(0)) = hash(d.p, u)
Base.length(d::BoolDistribution) = 2

@with_kw struct CryingBaby
    r_hungry::Float64 = -10.0
    r_feed::Float64 = -5.0
    r_sing::Float64 = -0.5
    p_become_hungry::Float64 = 0.1
    p_cry_when_hungry::Float64 = 0.8
    p_cry_when_not_hungry::Float64 = 0.1
    p_cry_when_hungry_in_sing::Float64 = 0.9
    γ::Float64 = 0.9
end

# CryingBaby = CryingBaby(-10.0, -5.0, -0.5, 0.1, 0.8, 0.1, 0.9, 0.9)

SATED = 1
HUNGRY = 2
FEED = 1
IGNORE = 2
SING = 3
CRYING = true
QUIET = false

CRYING_BABY_ACTION_COLORS = Dict(
    FEED => "pastelBlue",
    IGNORE => "pastelGreen",
    SING => "pastelRed"
)
CRYING_BABY_ACTION_NAMES = Dict(
    FEED => "feed",
    IGNORE => "ignore",
    SING => "sing",
)

n_states(::CryingBaby) = 2
n_actions(::CryingBaby) = 3
n_observations(::CryingBaby) = 2
discount(pomdp::CryingBaby) = pomdp.γ

ordered_states(::CryingBaby) = [SATED, HUNGRY]
ordered_actions(::CryingBaby) = [FEED, IGNORE, SING]
ordered_observations(::CryingBaby) = [CRYING, QUIET]

two_state_categorical(p1::Float64) = Categorical([p1, 1.0 - p1])

function transition(pomdp::CryingBaby, s::Int, a::Int)
    if a == FEED
        return two_state_categorical(1.0) # [1, 0]
    else
        if s == HUNGRY
            return two_state_categorical(0.0) # [0, 1]
        else
            # Did not feed when not hungry
            return two_state_categorical(1.0 - pomdp.p_become_hungry) # [1-p_become_hungry, p_become_hungry]
        end
    end
end

function observation(pomdp::CryingBaby, a::Int, s′::Int)
    if a == SING
        if s′ == HUNGRY
            return BoolDistribution(pomdp.p_cry_when_hungry_in_sing) # b.p=0.9
        else
            return BoolDistribution(0.0) # b.p=0
        end
    else # FEED or IGNORE
        if s′ == HUNGRY
            return BoolDistribution(pomdp.p_cry_when_hungry)
        else
            return BoolDistribution(pomdp.p_cry_when_not_hungry)
        end
    end
end

function reward(pomdp::CryingBaby, s::Int, a::Int)
    r = 0.0
    if s == HUNGRY
        r += pomdp.r_hungry # -10
    end
    if a == FEED
        r += pomdp.r_feed # -5
    elseif a == SING
        r += pomdp.r_sing # -0.5
    end
    return r
end

reward(pomdp::CryingBaby, b::Vector{Float64}, a::Int) = sum(reward(pomdp, s, a) * b[s] for s in ordered_states(pomdp))
# reward=(p_hungry*reward_hungry+p_sated*reward_sated)

function DiscretePOMDP(pomdp::CryingBaby; γ::Float64 = pomdp.γ)
    nS = n_states(pomdp)
    nA = n_actions(pomdp)
    nO = n_observations(pomdp)

    T = zeros(nS, nA, nS)
    R = Array{Float64}(undef, nS, nA)
    O = Array{Float64}(undef, nA, nS, nO)

    s_s = 1 #sated
    s_h = 2 #hungry

    a_f = 1 #feed
    a_i = 2 #ignore
    a_s = 3 #sing

    o_c = 1 #cry
    o_q = 2 #quiet

    T[s_s, a_f, :] = [1.0, 0.0] # T(sated | feed, (hungry or sated))
    T[s_s, a_i, :] = [1.0 - pomdp.p_become_hungry, pomdp.p_become_hungry]
    T[s_s, a_s, :] = [1.0 - pomdp.p_become_hungry, pomdp.p_become_hungry]
    T[s_h, a_f, :] = [1.0, 0.0]
    T[s_h, a_i, :] = [0.0, 1.0]
    T[s_h, a_s, :] = [0.0, 1.0]

    R[s_s, a_f] = reward(pomdp, s_s, a_f) # R(sated, feed)
    R[s_s, a_i] = reward(pomdp, s_s, a_i)
    R[s_s, a_s] = reward(pomdp, s_s, a_s)
    R[s_h, a_f] = reward(pomdp, s_h, a_f)
    R[s_h, a_i] = reward(pomdp, s_h, a_i)
    R[s_h, a_s] = reward(pomdp, s_h, a_s)

    O[a_f, s_s, :] = [observation(pomdp, a_f, s_s).p, 1 - observation(pomdp, a_f, s_s).p] # [O(cry|feed, sated), O(quiet|feed, sated)]
    O[a_f, s_h, :] = [observation(pomdp, a_f, s_h).p, 1 - observation(pomdp, a_f, s_h).p]
    O[a_i, s_s, :] = [observation(pomdp, a_i, s_s).p, 1 - observation(pomdp, a_i, s_s).p]
    O[a_i, s_h, :] = [observation(pomdp, a_i, s_h).p, 1 - observation(pomdp, a_i, s_h).p]
    O[a_s, s_s, :] = [observation(pomdp, a_s, s_s).p, 1 - observation(pomdp, a_s, s_s).p]
    O[a_s, s_h, :] = [observation(pomdp, a_s, s_h).p, 1 - observation(pomdp, a_s, s_h).p]

    return DiscretePOMDP(T, R, O, γ)
end

function POMDP(pomdp::CryingBaby; γ::Float64 = pomdp.γ)
    disc_pomdp = DiscretePOMDP(pomdp)
    return POMDP(disc_pomdp)
end

struct BabyPOMG
    babyPOMDP::CryingBaby
end

function MultiCaregiverCryingBaby()
    BabyPOMDP = CryingBaby()
    return BabyPOMG(BabyPOMDP)
end

n_agents(pomg::BabyPOMG) = 2

ordered_states(pomg::BabyPOMG) = [SATED, HUNGRY]
ordered_actions(pomg::BabyPOMG, i::Int) = [FEED, IGNORE, SING]
ordered_joint_actions(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_actions(pomg, i) for i in 1:n_agents(pomg)]...)))

n_actions(pomg::BabyPOMG, i::Int) = length(ordered_actions(pomg, i))
n_joint_actions(pomg::BabyPOMG) = length(ordered_joint_actions(pomg))

ordered_observations(pomg::BabyPOMG, i::Int) = [CRYING, QUIET]
ordered_joint_observations(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_observations(pomg, i) for i in 1:n_agents(pomg)]...)))

n_observations(pomg::BabyPOMG, i::Int) = length(ordered_observations(pomg, i))
n_joint_observations(pomg::BabyPOMG) = length(ordered_joint_observations(pomg))

function transition(pomg::BabyPOMG, s, a, s′) # a: joint actions [a1, a2]
    # Regardless, feeding makes the baby sated.
    if a[1] == FEED || a[2] == FEED
        if s′ == SATED
            return 1.0
        else
            return 0.0
        end
    else
        # If neither caretaker fed, then one of two things happens.
        # First, a baby that is hungry remains hungry.
        if s == HUNGRY
            if s′ == HUNGRY
                return 1.0
            else
                return 0.0
            end
            # Otherwise, it becomes hungry with a fixed probability.
        else
            probBecomeHungry = pomg.babyPOMDP.p_become_hungry
            if s′ == SATED
                return 1.0 - probBecomeHungry
            else
                return probBecomeHungry
            end
        end
    end
end

function joint_observation(pomg::BabyPOMG, a, s′, o)
    # If at least one caregiver sings, then both observe the result.
    if a[1] == SING || a[2] == SING
        # If the baby is hungry, then the caregivers both observe crying/silent together.
        if s′ == HUNGRY
            if o[1] == CRYING && o[2] == CRYING
                return pomg.babyPOMDP.p_cry_when_hungry_in_sing
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.babyPOMDP.p_cry_when_hungry_in_sing
            else
                return 0.0
            end
            # Otherwise the baby is sated, and the baby is silent.
        else
            if o[1] == QUIET && o[2] == QUIET
                return 1.0
            else
                return 0.0
            end
        end
        # Otherwise, the caregivers fed and/or ignored the baby.
    else
        # If the baby is hungry, then there's a probability it cries.
        if s′ == HUNGRY
            if o[1] == CRYING && o[2] == CRYING
                return pomg.babyPOMDP.p_cry_when_hungry
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.babyPOMDP.p_cry_when_hungry
            else
                return 0.0
            end
            # Similarly when it is sated.
        else
            if o[1] == CRYING && o[2] == CRYING
                return pomg.babyPOMDP.p_cry_when_not_hungry
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.babyPOMDP.p_cry_when_not_hungry
            else
                return 0.0
            end
        end
    end
end

function joint_reward(pomg::BabyPOMG, s, a)
    r = [0.0, 0.0]

    # Both caregivers do not want the child to be hungry.
    if s == HUNGRY
        r += [pomg.babyPOMDP.r_hungry, pomg.babyPOMDP.r_hungry] # both -10
    end

    # One caregiver prefers to feed.
    if a[1] == FEED
        r[1] += pomg.babyPOMDP.r_feed / 2.0 # -2.5
    elseif a[1] == SING
        r[1] += pomg.babyPOMDP.r_sing # -0.5
    end

    # One caregiver prefers to sing.
    if a[2] == FEED
        r[2] += pomg.babyPOMDP.r_feed # -5
    elseif a[2] == SING
        r[2] += pomg.babyPOMDP.r_sing / 2.0 # -0.25
    end

    # Note that caregivers only experience a cost if they do something.

    return r
end

joint_reward(pomg::BabyPOMG, b::Vector{Float64}, a) = sum(joint_reward(pomg, s, a) * b[s] for s in ordered_states(pomg))
# joint_reward(pomg, b, action)=b[sated]*joint_reward(pomg, sated, action)+b[hungry]*joint_reward(pomg, hungry, action)

function POMG(pomg::BabyPOMG)
    return POMG(
        pomg.babyPOMDP.γ, # discount factor
        vec(collect(1:n_agents(pomg))), # agents
        ordered_states(pomg), # state
        [ordered_actions(pomg, i) for i in 1:n_agents(pomg)], # joint action space
        [ordered_observations(pomg, i) for i in 1:n_agents(pomg)], # joint observation space
        (s, a, s′) -> transition(pomg, s, a, s′),  # Transition(s'|s, a)
        (a, s′, o) -> joint_observation(pomg, a, s′, o), # joint observation function
        (s, a) -> joint_reward(pomg, s, a) # Reward(s, a)
    )
end

struct POMGNashEquilibrium
    b # initial belief b[sated], b[hungry]
    d # depth of conditional plans
end

struct ConditionalPlan
    # can be represented by tree
    a # action to take at root
    subplans # dictionary mapping observations to subplans (sub-conditional plan)
end


ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]

function lookahead(𝒫::POMG, U, s, a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    u′ = sum(T(s, a, s′) * sum(O(a, s′, o) * U(o, s′) for o in 𝒪) for s′ in 𝒮)
    return R(s, a) + γ * u′
end

function evaluate_plan(𝒫::POMG, π, s)
    # compute utility of conditional plan 
    a = Tuple(πi() for πi in π)
    U(o, s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π, o)], s′)
    return isempty(first(π).subplans) ? 𝒫.R(s, a) : lookahead(𝒫, U, s, a) # equation (26.1) page 528
end

function utility(𝒫::POMG, b, π)
    # compute utility of policy π from initial state distibution b
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs * us for (bs, us) in zip(b, u)) # equation (26.2) page 528
end


function create_conditional_plans(𝒫, d)
    # create conditional plan with depth d from P::POMG
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
    end
    return Π
end

function expand_conditional_plans(𝒫, Π)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i])) for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end

function tensorform(𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    ℐ′ = eachindex(ℐ)
    𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
    R′ = [R(a) for a in joint(𝒜)]
    return ℐ′, 𝒜′, R′
end

struct NashEquilibrium end

function solve(M::NashEquilibrium, 𝒫::SimpleGame)
    # find nash equilibrum for SimpleGame
    ℐ, 𝒜, R = tensorform(𝒫)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[ℐ])
    @variable(model, π[i = ℐ, 𝒜[i]] ≥ 0)
    @NLobjective(model, Min,
        sum(U[i] - sum(prod(π[j, a[j]] for j in ℐ) * R[y][i]
                       for (y, a) in enumerate(joint(𝒜))) for i in ℐ))
    @NLconstraint(model, [i = ℐ, ai = 𝒜[i]],
        U[i] ≥ sum(
            prod(j == i ? (a[j] == ai ? 1.0 : 0.0) : π[j, a[j]] for j in ℐ)
            *
            R[y][i] for (y, a) in enumerate(joint(𝒜))))
    @constraint(model, [i = ℐ], sum(π[i, ai] for ai in 𝒜[i]) == 1)
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai] => value(π[i, ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end



joint(X) = vec(collect(Iterators.product(X...)))


function solve(M::POMGNashEquilibrium, 𝒫::POMG)
    # step 1: convert POMG to SimpleGame
    # step 2: find Nash Equilibrium of that SimpleGame
    ℐ, γ, b, d = 𝒫.ℐ, 𝒫.γ, M.b, M.d
    Π = create_conditional_plans(𝒫, d)
    U = Dict(π => utility(𝒫, b, π) for π in joint(Π))
    𝒢 = SimpleGame(γ, ℐ, Π, π -> U[π])
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

struct POMGDynamicProgramming
    b # initial belief
    d # depth of conditional plans
end

function solve(M::POMGDynamicProgramming, 𝒫::POMG)
    ℐ, 𝒮, 𝒜, R, γ, b, d = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ, M.b, M.d
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
        prune_dominated!(Π, 𝒫)
    end
    𝒢 = SimpleGame(γ, ℐ, Π, π -> utility(𝒫, b, π))
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

function prune_dominated!(Π, 𝒫::POMG)
    # prune any policy that is dominated by another policies
    done = false
    while !done
        done = true
        for i in shuffle(𝒫.ℐ)
            for πi in shuffle(Π[i])
                if length(Π[i]) > 1 && is_dominated(𝒫, Π, i, πi)
                    filter!(πi′ -> πi′ ≠ πi, Π[i])
                    done = false
                    break
                end
            end
        end
    end
end

function is_dominated(𝒫::POMG, Π, i, πi)
    # check if policy is dominated
    ℐ, 𝒮 = 𝒫.ℐ, 𝒫.𝒮
    jointΠnoti = joint([Π[j] for j in ℐ if j ≠ i])
    π(πi′, πnoti) = [j == i ? πi′ : πnoti[j > i ? j - 1 : j] for j in ℐ]
    Ui = Dict((πi′, πnoti, s) => evaluate_plan(𝒫, π(πi′, πnoti), s)[i]
              for πi′ in Π[i], πnoti in jointΠnoti, s in 𝒮)
    model = Model(Ipopt.Optimizer)
    @variable(model, δ)
    @variable(model, b[jointΠnoti, 𝒮] ≥ 0)
    @objective(model, Max, δ)
    @constraint(model, [πi′ = Π[i]],
        sum(b[πnoti, s] * (Ui[πi′, πnoti, s] - Ui[πi, πnoti, s])
            for πnoti in jointΠnoti for s in 𝒮) ≥ δ)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(δ) ≥ 0
end


multicaregiver_cryingbaby = MultiCaregiverCryingBaby() # return instance babyPOMG
pomg = POMG(multicaregiver_cryingbaby) # return POMG instance from babyPOMG instance

b = [0.8, 0.2] # initial state distribution, b[sated]=b[hungry]=0.5, we can set this to [0.8, 0.2]
d = 3 # depth of conditional plans

pomgDP = POMGDynamicProgramming(b, 5)
# pomgNash=POMGNashEquilibrium(b, d)
ans = solve(pomgDP, pomg)
print(ans)

vectorAns = []

function createVector!(c::ConditionalPlan, vectorAns, i)
    # if(length(vectorAns)<2*i-1)
    #     push!(vectorAns, [])
    # end
    # push!(vectorAns[2*i-1], c.a)
    # for (key, value) in c.subplans
    #     if(length(vectorAns)<2*i)
    #         push!(vectorAns, [])
    #     end
    #     push!(vectorAns[2*i], key)
    #     createVector!(value, vectorAns, i+1)
    # end
    if (length(vectorAns) < i)
        push!(vectorAns, [])
    end
    push!(vectorAns[i], c.a)
    for (key, value) in c.subplans
        createVector!(value, vectorAns, i + 1)
    end
end

function printSpace(n)
    for i = 1:n
        print(" ")
    end
end

function printVectorAns(vectorAns)
    powerOf2 = [64, 32, 16, 8, 4, 2]
    println("")
    for i = 1:length(vectorAns)
        printSpace(powerOf2[i] / 2 - 1)
        for j = 1:length(vectorAns[i])

            # printSpace(powerOf2[i]-1)
            item = vectorAns[i][j]
            if item == FEED
                print("F")
            elseif item == SING
                print("S")
            else
                print("I")
            end
            if j != length(vectorAns[i])
                printSpace(powerOf2[i] - 1)
            end
        end
        println("")
    end
end

for res in ans
    createVector!(res, vectorAns, 1)
    printVectorAns(vectorAns)
    empty!(vectorAns)
end



