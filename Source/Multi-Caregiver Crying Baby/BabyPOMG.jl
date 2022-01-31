include("CryingBaby.jl")

struct BabyPOMG
    babyPOMDP::CryingBaby
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