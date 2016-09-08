

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


