module Anamnesis
# TODO make DataFrames dependency conditional and remove DatasToolbox dependency
using DataFrames

# TODO implement kwargs



const METADATE_FILENAME = "metadata.jbin"
const FILENAME_RANDSTRING_LENGTH = 12
const ARGHASH_FLOAT_DIGITS = 12


abstract AbstractScribe

# these are scribes that are getting stored for macros
ScribeBox = Dict{Symbol,AbstractScribe}()


#=================================================================================
    Metadata Dictionary Format

The metadata dictionary has keys which are Vector{Any}s with the function name
followed by the hashed arguments.
=================================================================================#
metadataFileName(dir::String) = joinpath(dir, METADATE_FILENAME)
loadMetadata_raw(dir::String) = deserialize(metadataFileName(dir))

function loadMetadata(dir::String)
    filename = metadataFileName(dir)
    if isfile(filename)
        return loadMetadata_raw(dir)
    end
    Dict()  # if file doesn't exist, return empty Dict
end


function saveMetadata(dir::String, meta::Dict)
    serialize(joinpath(dir, METADATE_FILENAME), meta)
end

# this gets the metadata dict that would be stored in the file from a scribe
function metadata(s::AbstractScribe)
    meta = copy(s.vals)
    for (k, v) ∈ meta
        nv = Dict{Any,Any}(k_=>v_ for (k_, v_) ∈ v if k_ ≠ :val)  # store all but value
        meta[k] = nv
    end
    meta
end

function loadfile(filename::String)
    ext = convert(String, split(filename, '.')[end])
    if ext == ".feather"
        return featherRead(filename)
    end
    deserialize(filename)
end

#====================================================================================
    Scribes

These are the objects that store a function along with a means of memorizing it.

All data about function evaluations are stored in the `vals` field, which is a 
`Dict` of `Dict`s.  The reason for the nested dicts as opposed to multiple dicts
is to make these more extensible.
====================================================================================#
type VolatileScribe <: AbstractScribe
    f::Function
    vals::Dict
    name::Symbol

    VolatileScribe(f::Function, name::Symbol) = new(f, Dict(), name)
end
export VolatileScribe

# this is a separate type mainly for multiple dispatch reasons
type NonVolatileScribe <: AbstractScribe
    f::Function
    vals::Dict
    name::Symbol
    dir::String

    function NonVolatileScribe(f::Function, name::Symbol, dir::String)
        !isdir(dir) && mkdir(dir)
        new(f, loadMetadata(dir), name, dir)
    end
end
export NonVolatileScribe


# demotion to VolatileScribe retains `:file` entires; no reason to delete them
function VolatileScribe(s::NonVolatileScribe, dir::String)
    o = VolatileScribe(s.f, s.name)
    o.vals = s.vals
    o
end


# this is a promotion from a VolatileScribe to a NonVolatileScribe
function NonVolatileScribe(s::VolatileScribe, dir::String)
    o = NonVolatileScribe(s.f, s.name, dir)
    o.vals = s.vals
    # now we need to make files for everything that's missing
    for (k, v) ∈ o.vals
        y = v[:val]
        _save_eval!(o, v, k, y)
    end
    o
end



function _generate_filename_raw{T}(s::AbstractScribe, ::Type{T})
    filename = string(s.name, "_", randstring(FILENAME_RANDSTRING_LENGTH), ".jbin")
end
function _generate_filename_raw(s::AbstractScribe, ::Type{DataFrame})
    filename = string(s.name, "_", randstring(FILENAME_RANDSTRING_LENGTH), ".feather")
end

# this is a bad way of doing this but shouldn't matter
function _generate_filename{T}(s::AbstractScribe, ::Type{T})
    filename = _generate_filename_raw(s, T)
    while isfile(joinpath(s.dir, filename))
        filename = _generate_filename_raw(s, T)
    end
    filename
end


# this is is the implementation of "memoize"
function (s::VolatileScribe)(args...)
    key_args = hashArgs(args)
    dict = get(s.vals, key_args, Dict())
    if (:val ∈ keys(dict))
        y = dict[:val]
    else
        y = s.f(args...)
        dict[:val] = y
        s.vals[key_args] = dict
    end
    y
end


_save_eval(filename::String, y) = serialize(filename, y)
_save_eval(filename::String, y::DataFrame) = featherWrite(filename, y)

function _save_eval!(s::NonVolatileScribe, dict::Dict, key_args, y)
    filename = _generate_filename(s, typeof(y))
    dict[:val] = y
    dict[:file] = filename
    s.vals[key_args] = dict
    meta = metadata(s)  # TODO clean this shit up! shouldn't run all the time
    saveMetadata(s.dir, meta)  # for now we save metadata every time.  ugh
    _save_eval(joinpath(s.dir, filename), y)
end


# this is memoize with storing on disk
function (s::NonVolatileScribe)(args...)
    key_args = hashArgs(args)
    dict = get(s.vals, key_args, Dict())
    if (:val ∈ keys(dict))
        y = dict[:val]
    elseif (:file ∈ keys(dict))
        y = loadfile(joinpath(s.dir, dict[:file]))
        dict[:val] = y
        s.vals[key_args] = dict
    else
        y = s.f(args...)
        _save_eval!(s, dict, key_args, y)
    end
    y
end


# alias for call, useful when writing macros
execute!(s::AbstractScribe, args...) = s(args...)
export execute!


function forget!(s::AbstractScribe, args...)
    key_args = hashArgs(args) 
    y = get(s.vals[key_args], :val, nothing)
    delete!(s.vals, key_args)
    y
end

function forget!(s::NonVolatileScribe, args...)
    key_args = hashArgs(args)
    if (:file ∈ keys(s.vals[key_args]))
        rm(joinpath(s.dir, s.vals[key_args][:file]))  # want an error if missing
    end
    delete!(s.vals, key_args)
end
export forget!


purge(dir::String) = rm(dir, force=true, recursive=true)
purge!(s::NonVolatileScribe) = purge(s.dir)
export purge, purge!


function refresh!(s::AbstractScribe, args...)
    forget!(s, args...)
    s(args...)
end
export refresh!


arghash(a) = hash(a)
arghash(a::Number) = a
arghash(a::AbstractFloat) = round(a, ARGHASH_FLOAT_DIGITS)


function hashArgs(args::Union{Tuple,Vector})
    key_args = Vector{Any}(length(args))
    for (i, arg) ∈ enumerate(args)
        key_args[i] = arghash(arg)
    end
    key_args
end


# right now this is just an alias for FunctionScribe, will add to later
scribe(f::Function, name::Symbol) = VolatileScribe(f, name)
scribe(f::Function, name::Symbol, dir::String) = NonVolatileScribe(f, name, dir)
export scribe


macro scribe(f)
    fname = Expr(:quote, f)
    if f ∉ keys(ScribeBox)
        o = :(Anamnesis.ScribeBox[$fname] = Anamnesis.scribe($f, $fname))
    else
        o = :(Anamnesis.ScribeBox[$fname])
    end
    esc(o)
end

macro scribe(dir, f)
    fname = Expr(:quote, f)
    if f ∉ keys(ScribeBox)
        o = quote
            @assert typeof(dir) <: AbstractString "Directory must be a string."
            Anamnesis.ScribeBox[$fname] = Anamnesis.scribe($f, $fname, $dir)
        end
    else
        # check if existing needs to be promoted
        if !isa(ScribeBox[f], NonVolatileScribe)
            ScribeBox[f] = NonVolatileScribe(ScribeBox[f])
        end
        o = :(Anamnesis.ScribeBox[$fname])
    end
    esc(o)
end
export @scribe



#=========================================================================================
  <@anamnesis (and related)>
=========================================================================================#
function _anamnesis_getsymbols_call(expr::Expr)
    f = expr.args[1]
    args = Expr(:tuple, expr.args[2:end]...)
    nothing, f, args
end

function _anamnesis_getsymbols_assignment(expr::Expr)
    val = expr.args[1]    
    if expr.args[2].head ≠ :call
        throw(ArgumentError("@anamnesis argument must contain a function call."))
    end
    _val, f, args = _anamnesis_getsymbols_call(expr.args[2])
    val, f, args
end

function _anamnesis_getsymbols_(expr::Expr)
    if expr.head == :(=)
        return _anamnesis_getsymbols_assignment(expr)
    elseif expr.head == :call
        return _anamnesis_getsymbols_call(expr)
    end
end

# TODO make work for multiple assignments within a block
function _anamnesis_getsymbols(expr::Expr)
    if expr.head ∈ [:(=), :call]
        return _anamnesis_getsymbols_(expr)
    elseif expr.head == :block
        for arg ∈ expr.args
            if arg.head ∈ [:(=), :call]
                return _anamnesis_getsymbols_(arg)
            end
        end
    end
    throw(ArgumentError("@anamnesis argument must contain an assignment."))
end


macro anamnesis(refresh::Bool, dir, expr)
    val, f, args = _anamnesis_getsymbols(expr)
    fname = Expr(:quote, f)
    
    if f ∉ keys(ScribeBox)
        retrieveexpr = quote
            if length($dir) > 0
                Anamnesis.ScribeBox[$fname] = Anamnesis.scribe($f, $fname, $dir)
            else
                Anamnesis.ScribeBox[$fname] = Anamnesis.scribe($f, $fname)
            end
        end
    else  # in this case we check if we need to promote to NonVolatileScribe
        retrieveexpr = quote
            if length($dir) > 0
                if !isa(Anamnesis.ScribeBox[$fname], NonVolatileScribe)
                    Anamnesis.ScribeBox[$fname] = NonVolatileScribe(Anamnesis.ScribeBox[$fname], $dir)
                end
            end
        end
    end

    callsymb = refresh ? Symbol(:refresh!) : Symbol(:execute!)

    if val == nothing
        callexpr = :(Anamnesis.$callsymb(Anamnesis.ScribeBox[$fname], $args...))
    else
        callexpr = :($val = Anamnesis.$callsymb(Anamnesis.ScribeBox[$fname], $args...))
    end

    esc(quote
        $retrieveexpr
        $callexpr
    end)
end

macro anamnesis(dir, expr)
    esc(:(@anamnesis false $dir $expr))
end

macro anamnesis!(dir, expr)
    esc(:(@anamnesis true $dir $expr))
end

macro anamnesis(expr)
    esc(:(@anamnesis false "" $expr))
end

macro anamnesis!(expr)
    esc(:(@anamnesis true "" $expr))
end

export @anamnesis, @anamnesis!


end
