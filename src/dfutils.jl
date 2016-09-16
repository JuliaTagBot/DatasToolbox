

"""
    convertCol(df::DataFrame, col::Symbol, dtype::DataType)

Converts a column, possibly containing python objects, to a column with eltype `dtype`.
The column itself will be a `DataArray` with `NA` values inserted where Python
`None`s are found.  Note that this isn't terribly efficient because it has to check
for `None`s.
"""
function convertCol(df::DataFrame, col::Symbol, dtype::DataType)
    # return map(d -> convert(dtype, d), df[col])
    pyNone = pybuiltin("None")
    o = DataArray(Vector{dtype}(length(df[col])))
    for i in 1:length(o)
        if df[col][i] == pyNone
            o[i] = NA
        else
            o[i] = convert(dtype, df[col][i])
        end
    end
    return o
end
export convertCol


# TODO for now just checks the first element of each column
"""
    migrateTypes!(df::DataFrame)

Attempts to convert all columns of a dataframe to the proper type based on the Python
types found in it.  
**TODO** Right now this just checks the first elements in the column.  Also, will have
to change `ASCIIString` conversion to `String` for Julia v0.5.
"""
function migrateTypes!(df::DataFrame)
    @pyimport datetime
    @pyimport decimal
    for col in names(df)
        isinstance = pybuiltin("isinstance")
        pystr = pybuiltin("str")
        if isinstance(df[1, col], pystr)
            df[col] = convertCol(df, col, ASCIIString)
        # need to preserve the order of these because somehow date <: datetime
        elseif isinstance(df[1, col], datetime.datetime)
            df[col] = convertCol(df, col, DateTime)
        elseif isinstance(df[1, col], datetime.date)
            df[col] = convertCol(df, col, Date)
        # consider removing this, it's obscure and probably shouldn't always be checked
        elseif isinstance(df[1, col], decimal.Decimal)
            df[col] = convertCol(df, col, Float32)
        end
    end
end
export migrateTypes!


"""
Checks to see if the column is one of the types known to fuck up conversions.
If so, makes the appropriate changes.
"""
function _fixBadPyConversions(pydf::PyObject, col::AbstractString)
    @pyimport numpy as np
    # TODO there are probably more types like this that need to be handled
    pycol = get(pydf, col)
    if np.dtype(pycol) == np.dtype("<M8[ns]")
        newcol = pycol[:astype]("O")
        return newcol[:values]
    end
    # if not, just return the column as an array
    return pycol[:values]
end


"""
    convertPyDF(df::PyObject; migrate::Bool=true, fix_nones::Bool=true)

Converts a pandas dataframe to a Julia dataframe.  If `migrate` is true this will try
to properly assign types to columns.  If `fix_nones` is true, this will check for columns
which have eltype `PyObject` and convert them to have eltype `Any`, replacing all Python
`None`s with `NA`.
"""
function convertPyDF(df::PyObject; 
                     migrate::Bool=true,
                     fix_nones::Bool=true)
    jdf = DataFrame()
    for col in df[:columns]
        jdf[symbol(col)] = _fixBadPyConversions(df, col)
    end
    if migrate migrateTypes!(jdf) end
    # attempts to fix columns which wound up badly converted because of nones
    if fix_nones
        for col in names(jdf)
            if !(eltype(jdf[col]) == PyObject) continue end
            fixPyNones!(Any, jdf, col)
        end
    end
    return jdf
end
export convertPyDF


"""
    fixPyNones(dtype::DataType, a::DataArray)

Attempts to convert a `DataArray` to have eltype `dtype` while replacing all Python
`None`s with `NA`.
"""
function fixPyNones(dtype::DataType, a::DataArray)
    newa = @data([pyeval("x is None", x=x) ? NA : convert(dtype, x) for x in a])
    return newa
end
export fixPyNones


"""
    fixPyNones!(dtype::DataType, df::DataFrame, col::Symbol)

Attempts to convert a column of the dataframe to have eltype `dtype` while replacing all
Python `None`s with `NA`.
"""
function fixPyNones!(dtype::DataType, df::DataFrame, col::Symbol)
    df[col] = fixPyNones(dtype, df[col])
    return df
end
export fixPyNones!


"""
    fixPyNones!(df::DataFrame)

Attempts to automatically convert all columns of a dataframe to have eltype `Any` while
replacing all Python `None`s with `NA`.
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
    convert(dtype::Union{Type{Int32}, Type{Int64}}, a::DataArray)

This converts a column of floats that should have been ints, but got converted to
floats because it has missing values which were converted to NaN's.
The supplied `DataArray` should have eltype `Float32` or `Float64`.
"""
function convert(dtype::Union{Type{Int32}, Type{Int64}}, a::DataArray{Float32, 1})
    newa = @data([isnan(x) ? NA : convert(dtype, x) for x in a])
    return newa
end
export convert


function convert(dtype::Union{Type{Int32}, Type{Int64}}, a::DataArray{Float64, 1})
    newa = @data([isnan(x) ? NA : convert(dtype, x) for x in a])
    return newa
end


"""
    loadPickledDF(filename::AbstractString; migrate::Bool=true, fix_nones::Bool=true)

Loads a pickled python dataframe, converting it to a Julia dataframe using `convertPyDF`.
"""
function loadPickledDF(filename::AbstractString;
                       migrate::Bool=true,
                       fix_nones::Bool=true)
    f = pyeval("open(\"$filename\", \"rb\")")
    @pyimport pickle
    pydf = pickle.load(f)
    df = convertPyDF(pydf, migrate=migrate, fix_nones=fix_nones)
    return df
end
export loadPickledDF


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
    numericalCategories(otype::DataType, A::Array)

Converts a categorical variable into numerical values of the given type.

Returns the mapping as well as the new array, but the mapping is just an array
so it always maps to an integer
"""
function numericalCategories(otype::DataType, A::Array)
    mapping = sort!(unique(A))
    o = convert(Array{otype}, indexin(A, mapping))
    return o, mapping
end
# define for DataArray type
numericalCategories(otype::DataType, A::DataArray) = numericalCategories(otype, 
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
function numericalCategories!(otype::DataType, df::DataFrame, col::Symbol)
    df[symbol(string(col)*"_Orig")] = df[col]
    df[col], mapping = numericalCategories(otype, df[col])
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
function numericalCategories!(otype::DataType, df::DataFrame, cols::Array{Symbol})
    for col in cols
        numericalCategories!(otype, df, col)
    end
    return
end
export numericalCategories!


"""
    convertNulls!{T}(A::Array{T, 1}, newvalue::T)

Converts all null values (NaN's and Nullable()) to a particular value.
Note this has to check whether the type is Nullable.
"""
function convertNulls!{T}(A::Array{T, 1}, newvalue::T)
    if T <: Nullable
        for i in 1:length(A)
            if isnull(A[i])
                A[i] = newvalue
            end
        end
    end
    if T <: AbstractFloat
        for i in 1:length(A)
            if isnan(A[i])
                A[i] = newvalue
            end
        end
    end
    return A
end
export convertNulls!


"""
    convertNulls{T}(A::DataArray{T}, newvalue::T)

Converts all null vlaues (NA's, NaN's and Nullable()) to a particular value.
"""
function convertNulls{T}(A::DataArray{T}, newvalue::T)
    A = convert(Array, A, newvalue)
    convertNulls!(A, newvalue)
    return DataArray(A)
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
    dct = Dict([x=>y for (x, y) in kwargs])
    applyCatConstraints(dct, df)
end

export applyCatConstraints

