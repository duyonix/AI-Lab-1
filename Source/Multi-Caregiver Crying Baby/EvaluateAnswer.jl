

struct node
    p_hungry::Float64
    p_cry::Float64
end



function parent_node(i)
    return floor(Int8, (i + 1) / 2)
end

function is_previous_state_is_cry(i)
    return i % 2 == 1
end

function children_quiet(i)
    return i * 2 - 1
end

function children_cry(i)
    return i * 2
end

function construct_probability_tree!(C1, C2, probabilityTree)

    # C1 is policy caregiver 1, C2 is policy caregiver 2
    empty!(probabilityTree)
    n = length(C1)
    for i = 1:n
        if length(probabilityTree) < i
            push!(probabilityTree, [])
        end
        if i == 1
            temping::node = node(0.2, 0.26)
            # temping::node=node(0.5, 0.5) # evaluate with initial distribution [0.5, 0.5]
            push!(probabilityTree[1], temping)
        else
            m = length(C1[i])
            for j = 1:m
                parent = parent_node(j)
                if C1[i-1][parent] != FEED && C2[i-1][parent] != FEED
                    p_hungry = probabilityTree[i-1][parent].p_hungry + (1 - probabilityTree[i-1][parent].p_hungry) * 0.2
                    p_cry = p_hungry * 0.9 + (1 - p_hungry) * 0.1
                    temp1::node = node(p_hungry, p_cry)
                    push!(probabilityTree[i], temp1)
                else
                    temp2::node = node(0, 0.1)
                    push!(probabilityTree[i], temp2)
                end
            end
        end
    end
end

function computeExpectedUtility!(C, order, utility, probabilityTree)
    empty!(utility)
    n = length(C)
    for i = 1:n
        push!(utility, [])
    end
    for i = n:-1:1
        m = length(C[i])
        for j = 1:m
            if i == n
                u1 = probabilityTree[i][j].p_hungry * (-10)
                if C[i][j] == FEED
                    u1 = u1 - 5 / (3 - order)
                end
                push!(utility[i], u1)
            else
                u2 = probabilityTree[i][j].p_hungry * (-10)
                if C[i][j] == FEED
                    u2 = u2 - 5 / (3 - order)
                end
                u2 = u2 + probabilityTree[i][j].p_cry * utility[i+1][children_cry(j)]
                u2 = u2 + (1 - probabilityTree[i][j].p_cry) * utility[i+1][children_quiet(j)]
                push!(utility[i], u2)
            end
        end
    end
    return utility[1][1]
end

function evaluate_answer(C1, C2, C, order)
    probabilityTree = []
    utility = []
    # return true if C1 better than C2 and other caregiver's policy is C
    # order=1 ==> C1 or C2 is caregiver 1's policy
    construct_probability_tree!(C1, C, probabilityTree)
    E1 = computeExpectedUtility!(C1, order, utility, probabilityTree)
    # print("Expected utility C1: ")
    # println(E1)
    construct_probability_tree!(C2, C, probabilityTree)
    E2 = computeExpectedUtility!(C2, order, utility, probabilityTree)
    # print("Expected utility C2: ")
    # println(E2)
    return E1 >= E2
end