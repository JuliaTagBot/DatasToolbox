

"""
Serializes an object through a one line command.
"""
function serialize(filename::AbstractString, object)
    f = open(filename, "w")
    serialize(f, object)
    close(f)
    return nothing
end


"""
Deserializes a saved object through a one line command.
"""
function deserialize(filename::AbstractString)
    f = open(filename)
    o = deserialize(f)
    close(f)
    return o
end

