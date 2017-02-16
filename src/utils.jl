

"""
    serialize(filename::AbstractString, object)

Serializes an object and stores on the local file system to file `filename`.
This is quite similar to the functionality in `Base`, except that the default
`serialize` method requires an `IOStream` object instead of a file name, so
this eliminates an extra line of code.
"""
function serialize(filename::AbstractString, object)
    f = open(filename, "w")
    serialize(f, object)
    close(f)
    return nothing
end
export serialize


"""
    deserialize(filename::AbstractString)

Opens a file from the local file system and deserializes what it finds there.
This is quite similar to the functionality in `Base` except that the default 
`deserialize` method requires an `IOStream` object instead of a file name so this
eliminates an extra line of code.
"""
function deserialize(filename::AbstractString)
    f = open(filename)
    o = deserialize(f)
    close(f)
    return o
end
export deserialize


"""
    pwd2PyPath()

Adds the present working directory to the python path variable.
"""
function pwd2PyPath()
    unshift!(PyVector(pyimport("sys")["path"]), "")
end
export pwd2PyPath


# TODO make work with arbitrary variables
"""
    @pyslice slice

Gets a slice of a python object.  Does not shift indices.
"""
macro pyslice(slice)
    @assert slice.head == :ref
    obj = slice.args[1]
    slice_ = Expr(:quote, slice)
    o = quote
        pyeval(string($slice_), $obj=$obj)
    end
    return esc(o)
end
export @pyslice


"""
    getNormedHistogramData(X)

Very annoyingly Gadfly does not yet support normed histograms.

This function returns an ordered pair of vectors which can be 
fed to gadfly to create a normed histogram.  If the output
is `m, w` one can do
`plot(x=m, y=w, Geom.bar)` to create a histogram.
"""
function getNormedHistogramData{T <: Real}(X::Vector{T}; 
                                           nbins::Integer=StatsBase.sturges(length(X)))
    h = fit(Histogram, X, nbins=nbins)
    edges = h.edges[1]
    midpoints = Vector{Float64}(length(edges)-1) 
    for i in 1:length(midpoints)
        midpoints[i] = (edges[i+1] + edges[i])/2.0
    end
    width = midpoints[2] - midpoints[1]
    weights = h.weights./(sum(h.weights)*width)
    midpoints, weights
end

# NullableArrays version, just ignores nulls
function getNormedHistogramData{T <: Real}(X::NullableArray{T, 1};
                                           nbins::Integer=StatsBase.sturges(length(X)))
    X = dropnull(X)
    getNormedHistogramData(X, nbins=nbins)
end
export getNormedHistogramData


"""
    infast(x, collection)

Checks whether the object `x` is in `collection`. This is done efficiently by creating
hashes for the objects in `collection`.  This should only be used if `collection` is
large, as there is overhead in hashing and allocating.
"""
function infast{T}(x::T, collection::Vector{T})
    # not sure if this is the most efficient way to create the hash
    dict = Dict{T, Void}(c=>nothing for c ∈ collection)
    x ∈ keys(dict)
end
export infast


# TODO these really should be kept in another package
"""
    outer(A, B)

Performs the outer product of two tensors A_{i₁…iₙ}B_{j₁…jₙ}.

**TODO** Currently only implemented for A and B as vectors.
"""
function outer(A::Vector, B::Vector)
    C = Matrix(length(A), length(B))
    for j ∈ 1:size(C, 2), i ∈ 1:size(C, 1)
        C[i, j] = A[i]*B[j]
    end
    C
end

# there's no way around iterating over every element if A and B are dense
function outer{T<:AbstractSparseArray}(::Type{T}, A::Vector, B::Vector)
    C = spzeros(length(A), length(B))
    for j ∈ 1:size(C, 2), i ∈ 1:size(C, 1)
        C[i, j] = A[i]*B[j] # note that Julia keeps this element sparse if prod is 0.0
    end
    convert(T, C)
end
export outer


"""
    _info_expr(message, code)

Private method used by `@info`.
"""
function _info_expr(message::String, code::Union{Expr,Symbol})
    quote
        info(string("Executing ", $message, " ..."))
        $code
        info(string("Done executing ", $message, " ."))
    end
end

"""
    _infotime_expr(message, code)

Private method used by `@infotime`.
"""
function _infotime_expr(message::String, code::Union{Expr,Symbol})
    quote
        info(string("Executing ", $message, " ..."))
        @time $code
        info(string("Done executing ", $message, " ."))
    end
end

"""
    @info code

Executes code sandwhiched between informative info messages telling the user that the
code is being executed.
"""
macro info(message::String, code)
    esc(_info_expr(message, code))
end
macro info(code)
    message = string('`', code, '`')
    esc(_info_expr(message, code))
end
export @info


"""
    @infotime code

Executes code sandwhiched between informative info messages telling the user that the code
is being executed, while applying the `@time` macro to the code.
"""
macro infotime(message::String, code)
    esc(_infotime_expr(message, code))
end
macro infotime(code)
    message = string('`', code, '`')
    esc(_infotime_expr(message, code))
end
export @infotime


#=========================================================================================
This section is all for breaking a matrix into submatrices based on a class label.
=========================================================================================#
getDefaultComparator{T}(::Type{T}) = (==)
getDefaultComparator{T<:AbstractFloat}(::Type{T}) = (≈)
export getDefaultComparator


"""
    findboundaries(X[, ncol; comparator=DEFAULT, check_sort=true])

Finds the boundaries between distinct values in the sorted array `X`.  For example, if
`X = [1, 1, 1, 2, 2, 3, 4]`, `boundaries(X) == [1, 4, 6, 7]`.  This algorithm requires
the input array to be sorted in the column in which boundaries are determined.
Do `sort_array=false` in cases where the input array is already sorted

This algorithm only works properly if the input array is already sorted.  By default,
this function checks if the input array is sorted.  To skip this check, set 
`check_sort=false`.
"""
function findBoundaries{T}(X::Vector{T}; check_sort::Bool=true,
                           comparator=getDefaultComparator(T))
    if check_sort && !issorted(X)
        throw(ArgumentError("Trying to find boundaries on unsorted array."))
    end
    boundaries = Int64[1]
    for i ∈ 2:length(X)
        !comparator(X[i-1], X[i]) && push!(boundaries, i)
    end
    boundaries
end

function findBoundaries{T}(X::Matrix{T}, ncol::Integer; check_sort::Bool=true,
                           comparator=getDefaultComparator(T))
    findBoundaries(X[:,ncol], check_sort=check_sort, comparator=comparator)
end


"""
    findboundaries!(v[; comparator=DEFAULT])

This is the same as `findboundaries` except that it sorts the vector `v` first. 
See documentation for `findboundaries`.
"""
function findBoundaries!{T}(X::Vector{T}; comparator=getDefaultComparator(T))
    if sort_array
        X = sort!(X)
    end
    findBoundaries(X, comparator=comparator)
end


"""
    boundaryClassValues(X[, ncol,] boundaries)

Given `boundaries` between distinct values in column `ncol` of `Matrix` `X` (can also be a
`Vector`) (see documentation for `findBoundaries`), find the values of the distinct values.

This is essentially the same functionality as unique, except that it is faster since
boundaries have already been determined.
"""
function boundaryClassValues{T}(X::Matrix{T}, ncol::Integer, boundaries::Vector)
    [X[boundaries[i], ncol] for i ∈ 1:length(boundaries)]
end
function boundaryClassValues{T}(v::Vector{T}, boundaries::Vector)
    [X[boundaries[i]] for i ∈ 1:length(boundaries)]
end


"""
    findBoundariesVlaues(X[, ncol; check_sort=true, comparator=DEFAULT])

Returns `boundaries, classes` where `boundaries` is the output of `findBoundaries` and
`classes` is the output of `boundaryClassValues`.  Note that `X` should be sorted before
inputting it into this function.
"""
function findBoundariesValues{T}(X::Vector{T}; check_sort::Bool=true,
                                 comparator=getDefaultComparator(T))
    boundaries = findBoundaries(X, check_sort=check_sort, comparator=comparator)
    classes = boundaryClassValues(X, boundaries)
    boundaries, classes
end

function findBoundariesValues{T}(X::Matrix{T}, ncol::Integer; check_sort::Bool=true,
                                 comparator=getDefaultComparator(T))
    boundaries = findBoundaries(X, ncol, check_sort=check_sort, comparator=comparator)
    classes = boundaryClassValues(X, ncol, boundaries)
    boundaries, classes
end


"""
    findBoundaryDict(X[, ncol; check_sort=true, comparator=DEFAULT])

For a sorted array `X`, find the boundaries between distinct values of column number
`ncol`.  
"""
function findBoundaryDict{T}(X::Vector{T}; check_sort::Bool=true,
                             comparator=getDefaultComparator(T))
    boundaries, classes = findBoundariesValues(X, check_sort=check_sort,
                                               comparator=comparator)
    dict = Dict{T,Tuple{Int64,Int64}}();  sizehint!(dict, length(boundaries))
    for i ∈ 1:length(boundaries)
        dict[classes[i]] = (boundaries[i], get(boundaries, i+1, length(X)+1)-1)
    end
    dict
end

function findBoundaryDict{T}(X::Matrix{T}, ncol::Integer; check_sort::Bool=true,
                             comparator=getDefaultComparator(T))
    boundaries, classes = findBoundariesValues(X, ncol, check_sort=check_sort,
                                               comparator=comparator)
    dict = Dict{T,Tuple{Int64,Int64}}();  sizehint!(dict, length(boundaries))
    for i ∈ 1:length(boundaries)
        dict[classes[i]] = (boundaries[i], get(boundaries, i+1, size(X,1)+1)-1)
    end
    dict
end
export findBoundaryDict


_order_y(y::Vector, order::Vector) = y[order]
_order_y(y::Matrix, order::Vector) = y[order, :]

_get_y_interval(y::Vector, r::UnitRange) = y[r]
_get_y_interval(y::Matrix, r::UnitRange) = y[r, :]


"""
    subMatricesByClass(X[, y], ncol)

Breaks the arrays `X` and `y` (optional) into dictionaries of `class=>submatrix` 
pairs where `class` is one of the distinct values of the column `ncol` of `X` and 
`submatrix` is the range of rows of `X` or `y` where `X[:,ncol]` has the value `class`.

This function is intended for breaking up training and test sets to be used with sets of
different models.
"""
function subMatricesByClass{T,U}(X::Matrix{T}, y::U, ncol::Integer)
    if size(X,1) ≠ size(y,1)
        throw(ArgumentError("X and y must have same number of rows."))
    end
    order = sortperm(X[:, ncol])
    X = X[order, :]
    y = _order_y(y, order)
    bdict = findBoundaryDict(X, ncol, check_sort=false)
    Xdict = Dict{T,Matrix{T}}();  sizehint!(Xdict, length(bdict))
    ydict = Dict{T,U}();  sizehint!(ydict, length(bdict))
    for (k, v) ∈ bdict
        Xdict[k] = X[v[1]:v[2], :]
        ydict[k] = _get_y_interval(y, v[1]:v[2])
    end
    Xdict, ydict
end

function subMatricesByClass{T}(X::Matrix{T}, ncol::Integer)
    order = sortperm(X[:, ncol])
    X = X[order, :]
    bdict = findBoundaryDict(X, ncol, check_sort=false)
    Xdict = Dict{T,Matrix{T}}();  sizehint!(Xdict, length(bdict))
    for (k, v) ∈ bdict
        Xdict[k] = X[v[1]:v[2], :]
    end
    Xdict
end
export subMatricesByClass

