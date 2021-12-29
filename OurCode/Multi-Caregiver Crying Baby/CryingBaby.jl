using Parameters: @with_kw

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
ordered_observations(::CryingBaby) = [CRYING,QUIET]

two_state_categorical(p1::Float64) = Categorical([p1,1.0 - p1])

function transition(pomdp::CryingBaby, s::Int, a::Int)
    if a == FEED
        return two_state_categorical(1.0) # [1, 0]
    else
        if s == HUNGRY
            return two_state_categorical(0.0) # [0, 1]
        else
            # Did not feed when not hungry
            return two_state_categorical(1.0-pomdp.p_become_hungry) # [1-p_become_hungry, p_become_hungry]
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

reward(pomdp::CryingBaby, b::Vector{Float64}, a::Int) = sum(reward(pomdp,s,a)*b[s] for s in ordered_states(pomdp))
# reward=(p_hungry*reward_hungry+p_sated*reward_sated)

function DiscretePOMDP(pomdp::CryingBaby; γ::Float64=pomdp.γ)
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
    T[s_s, a_i, :] = [1.0-pomdp.p_become_hungry, pomdp.p_become_hungry]
    T[s_s, a_s, :] = [1.0-pomdp.p_become_hungry, pomdp.p_become_hungry]
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

function POMDP(pomdp::CryingBaby; γ::Float64=pomdp.γ)
    disc_pomdp = DiscretePOMDP(pomdp)
    return POMDP(disc_pomdp)
end