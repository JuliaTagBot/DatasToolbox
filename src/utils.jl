

"""
Serializes an object through a one line command.
"""
function serialize(filename::AbstractString, object)
    f = open(filename, "w")
    serialize(f, object)
    close(f)
    return nothing
end
export serialize


"""
Deserializes a saved object through a one line command.
"""
function deserialize(filename::AbstractString)
    f = open(filename)
    o = deserialize(f)
    close(f)
    return o
end
export deserialize


"""
Adds the present working directory to the python path variables.
"""
function pwd2PyPath()
    unshift!(PyVector(pyimport("sys")["path"]), "")
end
export pwd2PyPath


# TODO make work with arbitrary variables
"""
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
