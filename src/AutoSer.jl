module AutoSer
using DataFrames  # TODO make this a conditional dependancy


const AUTOSER_METADATA_FILE = "metadata.jbin"
const AUTOSER_RANDSTRING_LENGTH = 12

AUTOSER_ARGKEY_FLOATDIGITS = 12

# note that return types can be determined with the undocumented 
# function Core.Inference.return_type

# TODO fix random string collisions
# TODO handle kwargs
# TODO checking for float args
# TODO consider having autoser return a function


AUTOSER_DIRECTORY = ""


_autoser_loadmeta(dir::String) = deserialize(joinpath(dir, AUTOSER_METADATA_FILE))


# too hard to make this depend on arguments
function _autoser_generate_filename(f_name::Symbol, extension::String=".jbin")
    string(f_name, "_", randstring(AUTOSER_RANDSTRING_LENGTH), extension)
end

function _autoser_generate_filename(f_name::Symbol, ::Type{DataFrame})
    _autoser_generate_filename(f_name, ".feather")
end

function _autoser_generate_filename{T}(f_name::Symbol, ::Type{T})
    _autoser_generate_filename(f_name, ".jbin")
end


_autoser_serialize_result(dir::String, file::String, y) = serialize(joinpath(dir, file), y)
function _autoser_serialize_result(dir::String, file::String, y::DataFrame) 
    featherWrite(joinpath(dir,file), y)
end

# these are our various hashing schemes for arguments
# the loss in hashing floats is quite deliberate
argkey(arg) = hash(arg)
argkey(arg::Number) = arg
argkey(arg::AbstractFloat) = round(arg, AUTOSER_ARGKEY_FLOATDIGITS)
# TODO put a size limit on this
argkey{T<:AbstractFloat}(arg::Array{T}) = round(arg, AUTOSER_ARGKEY_FLOATDIGITS)


function _autoser_meta_key(f_name::Symbol, args::Tuple)
    key_args = Vector{Any}(length(args))
    for (i, arg) ∈ enumerate(args)
        key_args[i] = argkey(arg)
    end
    tuple([f_name; key_args]...)
end


function _autoser_check_meta(meta::Dict, f_name::Symbol, args::Tuple)
    meta_key = _autoser_meta_key(f_name, args)
    get(meta, meta_key, "")
end


function _autoser_save_eval(directory::String, meta::Dict, f::Function, f_name::Symbol, 
                            args::Tuple)
    meta_key = _autoser_meta_key(f_name, args)
    y = f(args...)
    f_file = _autoser_generate_filename(f_name, typeof(y))
    while isfile(joinpath(directory, f_file))  # ensure no name collisions
        f_file = _autoser_generate_filename(f_name, typeof(y))
    end
    meta[meta_key] = f_file  # for now we only store the filename
    serialize(joinpath(directory, AUTOSER_METADATA_FILE), meta)
    _autoser_serialize_result(directory, f_file, y)
    y
end


function _autoser_load_eval(directory::String, fname::String)
    ext = convert(String, split(fname, '.')[end])
    if ext == "feather"
        return featherRead(joinpath(directory, fname))
    end
    deserialize(joinpath(directory, fname))
end


"""
    autoser(dir, f, f_name, args...[; override=false])
    autoser(f, f_name, args...[; override=false, dir])

Determines if the value output by evaluating `f(args...)` has already been computed and 
stored in the directory `dir` and, if so, returns it without evaluating `f`.  If not,
`f` is evaluated and the value is stored in a binary file in the directory `dir`.  
The file will have a name which can be looked up in `dir/metadata.jbin` using the function
name and arguments.

Note that this determines if the function value is stored using the name `f_name` and calls
`hash` on all `args`.

If `override`, `f` will be evaluated and a new value will be written to disk regardless
of whether one was previously stored.
"""
function autoser(directory::String, f::Function, f_name::Symbol, args...; 
                 override::Bool=false)
    @assert length(directory) > 0 "Must set a directory for serialization files."
    if !isdir(directory)
        mkdir(directory)
        y = _autoser_save_eval(directory, Dict(), f, f_name, args)
    else
        metafile = joinpath(directory, AUTOSER_METADATA_FILE)
        meta = isfile(metafile) ? deserialize(metafile) : Dict()
        fname = _autoser_check_meta(meta, f_name, args)
        if override
            fname = joinpath(directory, fname)
            isfile(fname) && rm(fname)
            y = _autoser_save_eval(directory, meta, f, f_name, args)
        elseif length(fname) == 0
            y = _autoser_save_eval(directory, meta, f, f_name, args)
        else
            y = _autoser_load_eval(directory, fname)
        end
    end
    y
end

function autoser(f::Function, f_name::Symbol, args...; dir::String=AUTOSER_DIRECTORY,
                 override::Bool=false)
    autoser(dir, f, f_name, args..., override=override)
end

export autoser


function autoserDir(directory::String)
    global AUTOSER_DIRECTORY = directory
end
export autoserDir


# this requires an assignment expression
function _autoser_macro_getsymbols_(expr::Expr)
    val = expr.args[1]    
    if expr.args[2].head ≠ :call
        throw(ArgumentError("@autoser argument must contain a function call."))
    end
    f = expr.args[2].args[1]
    args = Expr(:tuple, expr.args[2].args[2:end]...)
    val, f, args
end


# TODO make work for multiple assignments within a block
function _autoser_macro_getsymbols(expr::Expr)
    if expr.head == :(=)
        return _autoser_macro_getsymbols_(expr)
    elseif expr.head == :block
        for arg ∈ expr.args
            if arg.head == :(=)
                return _autoser_macro_getsymbols(arg)
            end
        end
    end
    throw(ArgumentError("@autoser argument must contain an assignment."))
end


"""
    @autoser directory y = f(x...) true
    @autoser directory y = f(x...)
    @autoser y = f(x...)

When supplied with an assignment resulting from a function evaluation, this macro will 
determine if the value of `f(x...)` has already been computed and, if so, return the
value stored on disk in the directory `directory` without evaluating `f`.  If not,
this will evaluate `f` and store the value in a binary file in `directory`.  If additionally
the `Bool` value `true` is supplied, `f` will be evaluated and any previous value will be
overwritten regardless of whether a computed value is stored.

If the value returned by `f` is a `DataFrame` it will be serialized using the feather format.

Note that this macro uses the name of the function and the value of arguments to determine
if the function has already been evaluated.  Therefore, changing the name of the function
will cause `@autoser` to consider this an entirely new function.  Furthermore, `hash` is
called on the function arguments, so this can still be quite slow when arguments are data
types of a very large size.  It is also possible (though extremely unlikely) that arguments
can get confused because they have the same hash value.  (In the future this can be improved
on, but there will always be a trade-off if one wants to avoid actually serializing large
input values.)

Currently this does not handle floats in any special way, so it is quite possible for 
machine error to spoof this macro call.
"""
macro autoser(directory, expr, override::Bool)
    val, f, args = _autoser_macro_getsymbols(expr)
    f_name = Meta.quot(f)
    esc(quote
        autoserDir($directory)
        $val = AutoSer.autoser($directory, $f, $f_name, $args..., override=$override)
    end)
end

macro autoserDir(directory)
    esc(:(AutoSer.autoserDir($directory)))
end

export @autoserDir

macro autoser(directory, expr)
    esc(:(@autoser $directory $expr false))
end


"""
    @autoser! directory y = f(x...)
    @autoser! y = f(x...)

This is an alias for `@autoser .... true`.  See documentation for `@autoser`.
"""
macro autoser!(directory, expr)
    esc(:(@autoser $directory $expr true))
end

macro autoser(expr)
    esc(:(@autoser AutoSer.AUTOSER_DIRECTORY $expr false))
end

macro autoser!(expr)
    esc(:(@autoser AutoSer.AUTOSER_DIRECTORY $expr true))
end


export @autoser
export @autoser!



end


