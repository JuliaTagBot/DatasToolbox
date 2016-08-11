

"""
Returns the discrete difference.  First value will be NaN.
"""
function discreteDiff{T}(X::Array{T, 1})
    if !(T <: AbstractFloat)
        error("Array elements must be of type with NaN.")
    end
    o = Array{T, 1}(length(X))
    o[1] = NaN
    for i in 2:length(X)
        o[i] = X[i] - X[i-1]
    end
    return o
end
discreteDiff{T}(X::DataArray{T, 1}) = discreteDiff(convert(Array, X))
export discreteDiff

