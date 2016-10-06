

"""
    discreteDiff{T}(X::Array{T, 1})

Returns the discrete difference between adjacent elements of a time series.  So, 
for instance, if one has a time series ``y_{1},y_{2},\ldots,y_{N}`` this will return
a set of ``δ`` such that ``δ_{i} = y_{i+1} - y_{i}``.  The first element of the returned
array will be a `NaN`.
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
discreteDiff{T}(X::NullableArray{T, 1}) = discreteDiff(convert(Array, X))
export discreteDiff

