using Distributions
using CategoricalArrays

struct SetCategorical{S}
    elements::Vector{S}
    distr::Categorical
    function SetCategorical(elements::AbstractVector{S}) where {S}
        weights = ones(length(elements))

        # return SetCategorical with the probabilities of all elements are the same
        return new{S}(elements, Categorical(normalize(weights, 1)))
    end
    function SetCategorical(elements::AbstractVector{S}, weights::AbstractVector{Float64}) where {S}
        # calculate norm 1
        ℓ₁ = norm(weights, 1)

        # if all weights equal zero
        if ℓ₁ < 1.0e-6 || isinf(ℓ₁)
            return SetCategorical(elements)
        end

        # normalize the weight so that total(weight) = 1 and keep the weight ratio the same
        distr = Categorical(normalize(weights, 1))
        return new{S}(elements, distr)
    end
end

# return the element selected from the random process based on the distribution of each element   
Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]
Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]
function Distributions.pdf(D::SetCategorical, x)
    # sum = distr.p of x in D.elements, if not x => 0
    sum(e == x ? w : 0.0 for (e, w) in zip(D.elements, D.distr.p))
end