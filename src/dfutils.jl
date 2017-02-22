

"""
    convertPyColumn(pycol::PyObject)

Converts a column of a pandas array to a Julia `NullableArray`.
"""
function convertPyColumn(pycol::PyObject)::NullableArray
    nptype = pycol[:dtype][:kind]
    # list of numpy kinds can be found at 
    # http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    if nptype == "O"
        o = NullableArray([x == nothing ? Nullable() : x for x in pycol])
        return o
    elseif nptype == "M"
        # doubt there are subtle efficiency differences between these, but it's possible
        o = NullableArray{DateTime}(
            [isa(t, PyObject) ? Nullable{DateTime}() : Nullable{DateTime}(t) for t in pycol])
        # (o = NullableArray(Nullable{DateTime}[isa(t, PyObject) ? Nullable() : t for t in
        #                                        pycol]))
        return o
    else
        return NullableArray(pycol[:values])
    end
end
export convertPyColumn


function _inferColumnType(col::NullableArray; max_elements::Integer=100)::DataType
    for i in 1:max_elements
        isnull(col) ? continue : nothing
        thistype = typeof(get(col[i]))
        thistype == PyObject ? continue : nothing
        return thistype
    end
    return Any
end


function _fillNewCol!{T, U}(newcol::NullableArray{T}, oldcol::NullableArray{U})
    for i in 1:length(newcol)
        if isnull(oldcol[i])
            newcol[i] = Nullable()
        else
            newcol[i] = get(oldcol[i])
        end
    end
end


"""
    fixColumnTypes!(df)

Check to see if the dataframe `df` has any columns of type `Any` and attempt to convert
them to the proper types.  This can be called from `convertPyDF` with the option
`fixtypes`.
"""
function fixColumnTypes!(df::DataFrame)
    for col in names(df)
        if eltype(eltype(df[col])) ≠ Any continue end
        dtype = _inferColumnType(df[col])
        # I can't find any way around getting into these stupid loops
        newcol = NullableArray{dtype}(length(df[col]))
        _fillNewCol!(newcol, df[col])
        df[col] = newcol
    end
end
export fixColumnTypes!


"""
    convertPyDF(pydf[, fixtypes=true])

Converts a pandas dataframe to a Julia one.  

Note that it is difficult to infer the correct types of columns which contain references
to Python objects.  If `fixtypes`, this will attempt to convert any column with eltype
`Any` to the proper type.
"""
function convertPyDF(pydf::PyObject; fixtypes::Bool=true)::DataFrame
    df = DataFrame()
    for col in pydf[:columns]
        df[Symbol(col)] = convertPyColumn(get(pydf, PyObject, col))
    end
    if fixtypes fixColumnTypes!(df) end
    return df
end
export convertPyDF


"""
    fixPyNones(dtype, a)

Attempts to convert a `NullableArray` to have eltype `dtype` while replacing all Python
`None`s with `Nullable`.
"""
function fixPyNones{T}(::Type{T}, a::NullableArray)
    # exit silently if the array can't possibly hold Nones
    if !((eltype(a) == Any) | (eltype(a) == PyObject)) return end
    pyNone = pybuiltin("None")
    newa = NullableArray(x == pyNone ? Nullable() : convert(T, x) for x in a)
end
export fixPyNones


"""
    fixPyNones!(dtype::DataType, df::DataFrame, col::Symbol)

Attempts to convert a column of the dataframe to have eltype `dtype` while replacing all
Python `None`s with `Nullable()`.
"""
function fixPyNones!{T}(::Type{T}, df::DataFrame, col::Symbol)
    df[col] = fixPyNones(T, df[col])
    return df
end
export fixPyNones!


"""
    fixPyNones!(df::DataFrame)

Attempts to automatically convert all columns of a dataframe to have eltype `Any` while
replacing all Python `None`s with `Nullable()`.
"""
function fixPyNones!(df::DataFrame)
    for col in names(df)
        if eltype(df[col]) == PyObject
            fixPyNones!(Any, df, col)
        end
    end
end
export fixPyNones!


"""
    convert(dtype, a)

This converts a column of floats that should have been ints, but got converted to
floats because it has missing values which were converted to NaN's.
The supplied `NullableArray` should have eltype `Float32` or `Float64`.
"""
function convert(dtype::Union{Type{Int32}, Type{Int64}}, a::NullableArray{Float32})
    NullableArray([isnan(x) ? Nullable() : convert(dtype, x) for x in a])    
end
export convert


function convert(dtype::Union{Type{Int32}, Type{Int64}}, a::NullableArray{Float64, 1})
    newa = NullableArray([isnan(x) ? Nullable() : convert(dtype, x) for x in a])
end


"""
    unpickle([dtype,] filename[, fixtypes=true])

Deserializes a python pickle file and returns the object it contains.
Additionally, if `DataFrame` is given as the first argument, will
attempt to convert the object to a Julia dataframe with the flag
`fixtypes` (see `convertPyDF`).
"""
function unpickle(filename::String)::PyObject
    f = py"open($filename, 'rb')"
    pyobj = PyPickle[:load](f)
end

function unpickle(::Type{DataFrame}, filename::AbstractString;
                  fixtypes::Bool=true)::DataFrame
    f = py"open($filename, 'rb')"
    # TODO it may be more efficient to create this from a dictionary than to convert
    pydf = pycall(PyPickle[:load], PyObject, f)
    df = convertPyDF(pydf, fixtypes=fixtypes)
end
export unpickle


"""
    pickle(filename, object)

Converts the provided object to a PyObject and serializes it in
the python pickle format.  If the object provided is a `DataFrame`,
this will first convert it to a pandas dataframe.
"""
function pickle(filename::String, object::Any)
    pyobject = PyObject(object)
    f = py"open($filename, 'wb')"
    PyPickle[:dump](pyobject, f)
end

function pickle(filename::String, df::DataFrame)
    pydf = pandas(df)
    f = py"open($filename, 'wb')"
    PyPickle[:dump](pydf, f)
end
export pickle


"""
    shuffle!(df::DataFrame)

Shuffles a dataframe in place.
"""
function shuffle!(df::DataFrame)
    permutation = shuffle(collect(1:size(df)[1]))
    tdf = copyColumns(df)
    for i in 1:length(permutation)
        df[i, :] = tdf[permutation[i], :]
    end
    return df
end
export shuffle!


"""
    numericalCategories(otype, A)

Converts a categorical variable into numerical values of the given type.

Returns the mapping as well as the new array, but the mapping is just an array
so it always maps to an integer
"""
function numericalCategories{T}(::Type{T}, A::Array)
    mapping = sort!(unique(A))
    o = convert(Array{T}, indexin(A, mapping))
    return o, mapping
end
# define for NullableArray type
numericalCategories{T}(::Type{T}, A::NullableArray) = numericalCategories(T, 
        convert(Array, A))
export numericalCategories


"""
    getDefaultCategoricalMapping(A::Array)

Gets the default mapping of categorical variables which would be returned by
numericalCategories.
"""
function getDefaultCategoricalMapping(A::Array)
    return sort!(unique(A))
end
export getDefaultCategoricalMapping


"""
    numericalCategories!(otype::DataType, df::DataFrame, col::Symbol)

Converts a categorical value in a column into a numerical variable of the given
type.

Returns the mapping.
"""
function numericalCategories!{T}(::Type{T}, df::DataFrame, col::Symbol)
    df[Symbol(string(col)*"_Orig")] = df[col]
    df[col], mapping = numericalCategories(T, df[col])
    return mapping
end
export numericalCategories!


"""
    numericalCategories!(otype::DataType, df::DataFrame, cols::Array{Symbol}) 

Converts categorical variables into numerical values for multiple columns in a
dataframe.  

**TODO** For now doesn't return mapping, may have to implement some type of 
mapping type.
"""
function numericalCategories!{T}(::Type{T}, df::DataFrame, cols::Array{Symbol})
    for col in cols
        numericalCategories!(T, df, col)
    end
    return
end
export numericalCategories!


"""
    convertNulls!{T}(A::Array{T, 1}, newvalue::T)

Converts all null values (NaN's and Nullable()) to a particular value.
Note this has to check whether the type is Nullable.
"""
function convertNulls!{T <: AbstractFloat}(A::Vector{T}, newvalue::T)
    for i in 1:length(A)
        if isnan(A[i])
            A[i] = newvalue
        end
    end
    return A
end

function convertNulls!{T <: Nullable}(A::Vector{T}, newvalue::T)
    for i in 1:length(A)
        if isnull(A[i])
            A[i] = newvalue
        end
    end
    return A
end

export convertNulls!


"""
    convertNulls{T}(A, newvalue)

Converts all null vlaues (NaN's and Nullable()) to a particular value.
This is a wrapper added for sake of naming consistency.
"""
function convertNulls{T}(A::NullableArray{T}, newvalue::T)
    convert(Array, A, newvalue)
end
export convertNulls


"""
    convertNulls!(df::DataFrame, cols::Vector{Symbol}, newvalue::Any)

Convert all null values in columns of a DataFrame to a particular value.

There is also a method for passing a single column symbol, not as a vector.
"""
function convertNulls!(df::DataFrame, cols::Vector{Symbol}, newvalue::Any)
    for col in cols
        df[col] = convertNulls(df[col], newvalue)
    end
    return
end
convertNulls!(df::DataFrame, col::Symbol, newvalue) = convertNulls!(df, [col], newvalue)
export convertNulls!


"""
    copyColumns(df::DataFrame)

The default copy method for dataframes only copies one level deep, so basically it stores
an array of columns.  If you assign elements of individual (column) arrays then, it can
make changes to references to those arrays that exist elsewhere.

This method instead creates a new dataframe out of copies of the (column) arrays.

This is not named copy due to the fact that there is already an explicit copy(::DataFrame)
implementation in dataframes.

Note that deepcopy is recursive, so this is *NOT* the same thing as deepcopy(df), which 
copies literally everything.
"""
function copyColumns(df::DataFrame)
    ndf = DataFrame()
    for col in names(df)
        ndf[col] = copy(df[col])
    end
    return ndf
end
export copyColumns


# this may be very inefficient due to the way Julia type optimization works
"""
    applyCatConstraints(dict, df[, kwargs])

Returns a copy of the dataframe `df` with categorical constraints applied.  `dict` should 
be a dictionary with keys equal to column names in `df` and values equal to the categorical
values that column is allowed to take on.  For example, to select gauge bosons we can
pass `Dict(:PID=>[i for i in 21:24; -24])`.  Alternatively, the values in the dictionary
can be functions which return boolean values, in which case the returned dataframe will
be the one with column values for which the functions return true.

Note that this requires that the dictionary values are either `Vector` or `Function` 
(though one can of course mix the two types).

Alternatively, instead of passing a `Dict` one can pass keywords, for example
`applyCatConstraints(df, PID=[i for i in 21:24; -24])`.
"""
function applyCatConstraints(dict::Dict, df::DataFrame)
    constr = Bool[true for i in 1:size(df)[1]]
    for (col, values) in dict
        constr &= if typeof(values) <: Vector
            convert(BitArray, map(x -> x ∈ values, df[col]))
        elseif typeof(values) <: Function
            convert(BitArray, map(values, df[col]))
        else
            throw(ArgumentError("Constraints must be either vectors or functions."))
        end
    end
    return df[constr, :]
end

function applyCatConstraints(df::DataFrame; kwargs...)
    dct = Dict(kwargs)
    applyCatConstraints(dct, df)
end
export applyCatConstraints


"""
    pandas(df)

Convert a dataframe to a pandas pyobject.
"""
function pandas(df::DataFrame)::PyObject
    pydf = pycall(PyPandas[:DataFrame], PyObject)
    for col in names(df)
        pycol = [isnull(x) ? nothing : get(x) for x in df[col]]
        set!(pydf, string(col), pycol)
        # convert datetime to proper numpy type
        if eltype(eltype(df[col])) == DateTime
            set!(pydf, string(col), 
                 get(pydf, string(col))[:astype]("<M8[ns]"))
        end
    end
    return pydf
end
export pandas


# helper function for constrain
function _colconstraints!{T}(col::NullableVector{T}, bfunc::Function, keep::Vector{Bool})
    for i ∈ 1:length(keep)
        if !isnull(col[i])
            keep[i] &= bfunc(get(col[i]))
        else
            keep[i] = false
        end
    end
    keep
end


# this is for fixing slowness due to bad dispatching
# performance is better, but it's still slow
function _dispatchConstrainFunc!(f::Function, mask::Vector{Bool},
                                 keep::BitArray, cols::NullableVector...)
    ncols = length(cols)
    # for some reason this completely fixes the performance issues
    # still don't completely understand why
    get_idx(i, j) = get(cols[j][i])
    get_args(i) = (get_idx(i, j) for j ∈ 1:ncols)
    # all arg cols are same length
    for i ∈ 1:length(keep)
        if !mask[i] 
            keep[i] = false
        else
            # this is inexplicably slow
            # note that this is even true for the generator alone; not a dispatch issue
            # keep[i] = f((get(col[i]) for col ∈ cols)...)
            keep[i] = f(get_args(i)...)
        end
    end
end

"""
    constrain(df, dict)
    constrain(df, kwargs...)
    constrain(df, cols, func)

Returns a subset of the dataframe `df` for which the column `key` satisfies 
`value(df[i, key]) == true`.  Where `(key, value)` are the pairs in `dict`.  
Alternatively one can use keyword arguments instead of a `Dict`.

Also, one can pass a function the arguments of which are elements of columns specified
by `cols`.
"""
function constrain{K<:Symbol, V<:Function}(df::AbstractDataFrame, constraints::Dict{K, V})::DataFrame
    keep = ones(Bool, size(df, 1))
    for (col, bfunc) ∈ constraints
        _colconstraints!(df[col], bfunc, keep)
    end
    df[keep, :]
end

function constrain{K, V<:Array}(df::AbstractDataFrame, constraints::Dict{K, V})::DataFrame
    newdict = Dict(k=>(x -> x ∈ v) for (k, v) ∈ constraints)
    constrain(df, newdict)
end

constrain(df::AbstractDataFrame; kwargs...) = constrain(df, Dict(kwargs))

function constrain(df::AbstractDataFrame, cols::Vector{Symbol}, f::Function)
    keep = BitArray(size(df, 1))
    _dispatchConstrainFunc!(f, complete_cases(df[cols]), keep, (df[col] for col ∈ cols)...)
    df[keep, :]
end
export constrain


# this is a helper function to @constrain
# TODO fix this so it works for variables used multiple times
function _checkConstraintExpr!(expr::Expr, dict::Dict)
    # the dictionary keys are the column names (as :(:col)) and the values are the symbols
    for (idx, arg) ∈ enumerate(expr.args)
        if isa(arg, QuoteNode)
            newsym = gensym()
            dict[Meta.quot(arg.value)] = newsym  
            expr.args[idx] = newsym
        elseif  isa(arg, Expr) && arg.head == :quote
            newsym = gensym()
            dict[Meta.quot(arg.args[1])] = newsym
            expr.args[idx] = newsym 
        elseif isa(arg, Expr)
            _checkConstraintExpr!(expr.args[idx], dict)
        end
    end
end

# TODO this is still having performance issues
"""
    @constrain(df, expr)

Constrains the dataframe to rows for which `expr` evaluates to `true`.  `expr` should
specify columns with column names written as symbols.  For example, to do `(a ∈ A) > M`
one should write `:A .> M`.
"""
macro constrain(df, expr)
    dict = Dict()
    _checkConstraintExpr!(expr, dict)
    cols = collect(keys(dict))
    vars = [dict[k] for k ∈ cols]
    cols_expr = Expr(:vect, cols...)
    fun_name = gensym()
    o = quote
        function $fun_name($(vars...)) 
            $expr
        end
        constrain($df, $cols_expr, $fun_name)
    end
    esc(o)
end
export @constrain


"""
    makeTestDF(dtypes...; nrows=10^4)

Creates a random dataframe with columns of types specified by `dtypes`.  This is useful
for testing various dataframe related functionality.
"""
function makeTestDF(dtypes::DataType...; nrows::Integer=10^4,
                    names::Vector{Symbol}=Symbol[])::DataFrame
    df = DataFrame()
    for (idx, dtype) in enumerate(dtypes)
        col = Symbol(string(dtype)*string(idx))
        if dtype <: Real
            df[col] = rand(dtype, nrows)
        elseif dtype <: AbstractString
            df[col] = [randstring(rand(8:16)) for i in 1:nrows]
        elseif dtype <: Dates.TimeType
            df[col] = [dtype(now()) + Dates.Day(i) for i in 1:nrows]
        elseif dtype <: Symbol
            df[col] = [Symbol(randstring(rand(4:12))) for i in 1:nrows]
        end
    end
    if length(names) > 0
        names!(df, names)
    end
    return df
end
export makeTestDF


"""
    nans2nulls(col)
    nans2nulls(df, colname)

Converts all `NaN`s appearing in the column to `Nullable()`.  The return
type is `NullableArray`, even if the original type of the column is not.
"""
function nans2nulls{T}(col::NullableArray{T})::NullableArray
    # this is being done without lift because of bugs in NullableArrays
    # map(x -> (isnan(x) ? Nullable{T}() : x), col, lift=true)
    map(col) do x
        if !isnull(x) && isnan(get(x))
            return Nullable{T}()
        end
        return x
    end
end

function nans2nulls(col::Vector)::NullableArray
    col = convert(NullableArray, col)
    nans2nulls(col)
end

function nans2nulls(df::DataFrame, col::Symbol)::NullableArray
    nans2nulls(df[col])
end
export nans2nulls


"""
    featherWrite(filename, df[, overwrite=false])

A wrapper for writing dataframes to feather files.  To be used while Feather.jl package
is in development.

If `overwrite`, this will delete the existing file first (an extra step taken to avoid some
strange bugs).
"""
function featherWrite(filename::AbstractString, df::DataFrame;
                      overwrite::Bool=false)::Void
    if isfile(filename)
        if !overwrite
            # TODO why is this not printing the string correctly
            throw(SystemError("File already exists.  Use overwrite=true."))
        end
        rm(filename)     
    end
    Feather.write(filename, df)
    return nothing
end
export featherWrite


"""
    convertWeakRefStrings(df)
    convertWeakRefStrings!(df)

Converts all columns with eltype `Nullable{WeakRefString}` to have eltype `Nullable{String}`.
`WeakRefString` is a special type of string used by the feather package to improve deserialization
performance.

Note that this will no longer be necessary in Julia 0.6.
"""
function convertWeakRefStrings(df::AbstractDataFrame)
    odf = DataFrame()
    for col ∈ names(df)
        if eltype(df[col]) <: Nullable{Feather.WeakRefString{UInt8}}
            odf[col] = convert(NullableVector{String}, df[col])
        else
            odf[col] = df[col]
        end
    end
    odf
end
export convertWeakRefStrings
function convertWeakRefStrings!(df::AbstractDataFrame)
    for col ∈ names(df)
        if eltype(df[col]) <: Nullable{Feather.WeakRefString{UInt8}}
            df[col] = convert(NullableVector{String}, df[col])
        end
    end
    df
end
export convertWeakRefStrings!


"""
    featherRead(filename[; convert_strings=true])

A wrapper for reading dataframes which are saved in feather files.  The purpose of this
wrapper is primarily for converting `WeakRefString` to `String`.  This will no longer
be necessary in Julia 0.6.
"""
function featherRead(filename::AbstractString; convert_strings::Bool=true)::DataFrame
    df = Feather.read(filename)
    if convert_strings
        convertWeakRefStrings!(df)
    end
    df
end
export featherRead


"""
## DatasToolbox
`DatasToolbox` provides the following new constructors for `Dict`:

    Dict(keys, values)
    Dict(df, keycol, valcol)

One can provide `Dict` with (equal length) vector arguments.  The first
vector provides a list of keys, while the second provides a list of values.
If the vectors are `NullableVector`, only key, value pairs with *both* their
elements non-null will be added.
"""
function Dict{K, V}(keys::Vector{K}, values::Vector{V})::Dict
    @assert length(keys) == length(values) ("Vectors for constructing 
                                             Dict must be of equal length.")
    Dict(k=>v for (k, v) ∈ zip(keys, values))
end

function Dict{K, V}(keys::NullableVector{K}, values::NullableVector{V})::Dict
    @assert length(keys) == length(values) ("Vectors for constructing
                                             Dict must be of equal length.")
    dict = Dict{K, V}()
    size_ = sum(!isnull(k) && !isnull(v) ? 1 : 0 for (k, v) ∈ zip(keys, values))::Integer
    sizehint!(dict, size_)
    # we only insert pairs if both values are not null
    for (k, v) ∈ zip(keys, values)
        if !isnull(k) && !isnull(v)
            dict[get(k)] = get(v)
        end
    end
    return dict
end

function Dict(df::DataFrame, keycol::Symbol, valcol::Symbol)::Dict
    Dict(df[keycol], df[valcol])
end
export Dict


"""
    getCategoryVector(A, vals[, dtype])

Get a vector which is 1 for each `a ∈ A` that satisfies `a ∈ vals`, and 0 otherwise.
If `A` is a `NullableVector`, any null elements will be mapped to 0.

Optionally, one can specify the datatype of the output vector.
"""
function getCategoryVector{T, U}(A::Vector{T}, vals::Vector{T}, ::Type{U}=Int64)
    # this is for efficiency
    valsdict = Dict{T, Void}(v=>nothing for v ∈ vals)
    Vector{U}([a ∈ keys(valsdict) for a ∈ A])
end

function getCategoryVector{T, U}(A::NullableVector{T}, vals::Vector{T}, ::Type{U}=Int64)
    # this is for efficiency
    valsdict = Dict{T, Void}(v=>nothing for v ∈ vals)
    o = map(a -> a ∈ keys(valsdict), A, lift=true)
    # these nested converts are the result of incomplete NullableArrays interface
    convert(Array{U}, convert(Array, o, 0))
end

function getCategoryVector{T, U}(A::Vector{T}, val::T, ::Type{U}=Int64)
    getCategoryVector(A, [val], U)
end

function getCategoryVector{T, U}(A::NullableVector{T}, val::T, ::Type{U}=Int64)
    getCategoryVector(A, [val], U)
end

function getCategoryVector{U}(df::AbstractDataFrame, col::Symbol, vals::Vector, ::Type{U}=Int64)
    getCategoryVector(df[col], vals, U)
end

function getCategoryVector{U}(df::AbstractDataFrame, col::Symbol, val, ::Type{U}=Int64)
    getCategoryVector(df[col], [val], U)
end
export getCategoryVector


"""
    getUnwrappedColumnElTypes(df[, cols=[]])

Get the element types of columns in a dataframe.  If the element types are `Nullable`, 
instead give the `eltype` of the `Nullable`.  If `cols=[]` this will be done for
all columns in the dataframe.
"""
function getUnwrappedColumnElTypes(df::DataFrame, cols::Vector{Symbol}=Symbol[])
    if length(cols) == 0
        cols = names(df)
    end
    [et <: Nullable ? eltype(et) : et for et ∈ eltypes(df[cols])]
end
export getUnwrappedColumnElTypes


"""
    getMatrixDict([T,] df, keycols, datacols)

Gets a dictionary the keys of which are the keys of a groupby of `df` by the columns
`keycols` and the values of which are the matrices produced by taking `sdf[datacols]`
of each `SubDataFrame` `sdf` in the groupby.  Note that the keys are always tuples
even if `keycols` only has one element.

If a type `T` is provided, the output matrices will be of type `Matrix{T}`.
"""
function getMatrixDict(df::DataFrame, keycols::Vector{Symbol}, datacols::Vector{Symbol})
    keycoltypes = getUnwrappedColumnElTypes(df, keycols)
    dict = Dict{Tuple{keycoltypes...},Matrix}()
    for sdf ∈ groupby(df, keycols)
        key = tuple(convert(Array{Any}, sdf[1, keycols])...)
        dict[key] = convert(Array, sdf[datacols])
    end
    dict
end

function getMatrixDict{T}(::Type{T}, gdf::GroupedDataFrame, keycols::Vector{Symbol},
                          datacols::Vector{Symbol})
    keycoltypes = getUnwrappedColumnElTypes(gdf.parent, keycols)
    dict = Dict{Tuple{keycoltypes...},Matrix{T}}()
    for sdf ∈ gdf
        key = tuple(convert(Array{Any}, sdf[1, keycols])...)
        dict[key] = convert(Array{T}, sdf[datacols])
    end
    dict
end

function getMatrixDict{T}(::Type{T}, gdf::GroupedDataFrame, keycols::Vector{Symbol},
                          Xcols::Vector{Symbol}, ycols::Vector{Symbol})
    keycoltypes = getUnwrappedColumnElTypes(gdf.parent, keycols)
    Xdict = Dict{Tuple{keycoltypes...},Matrix{T}}()
    ydict = Dict{Tuple{keycoltypes...},Matrix{T}}()
    for sdf ∈ gdf
        key = tuple(convert(Array{Any}, sdf[1, keycols])...)
        Xdict[key] = convert(Array{T}, sdf[Xcols])
        ydict[key] = convert(Array{T}, sdf[ycols])
    end
    Xdict, ydict
end

function getMatrixDict{T}(::Type{T}, df::DataFrame, keycols::Vector{Symbol},
                          datacols::Vector{Symbol})
    getMatrixDict(T, groupby(df, keycols), keycols, datacols)
end

# this version is used by grouped dataframe
function getMatrixDict{T}(::Type{T}, df::DataFrame, keycols::Vector{Symbol},
                          Xcols::Vector{Symbol}, ycols::Vector{Symbol})
    getMatrixDict(T, groupby(df, keycols), keycols, Xcols, ycols)
end

export getMatrixDict


