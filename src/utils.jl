

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
    @info code

Executes code sandwhiched between informative info messages telling the user that the
code is being executed.
"""
macro info(code)
    code_string = string(code)
    code_string = code_string[1:min(32, length(code_string))]
    esc(quote
        codestring = $code_string
        info("Executing `$codestring` ...")
        $code
        info("Done executing `$codestring` ...")
    end)
end
export @info


"""
    @infotime code

Executes code sandwhiched between informative info messages telling the user that the code
is being executed, while applying the `@time` macro to the code.
"""
macro infotime(code)
    code_string = string(code)
    code_string = code_string[1:min(32, length(code_string))]
    esc(quote
        codestring = $code_string
        info("Executing `$codestring` ...")
        @time $code
        info("Done executing `$codestring` ...")
    end)
end
export @infotime


