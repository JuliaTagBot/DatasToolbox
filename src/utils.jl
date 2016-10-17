

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


