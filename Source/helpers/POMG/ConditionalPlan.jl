struct ConditionalPlan
    # can be represented by tree
    a # action to take at root
    subplans # dictionary mapping observations to subplans (sub-conditional plan)
end

ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]