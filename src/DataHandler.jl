
abstract AbstractDH{T} <: Any
export AbstractDH


"""
Type for handling datasets.  This is basically a wrapper for a dataframe with methods
for splitting it into training and test sets and creating input and output numerical
arrays.  It is intended that most reformatting of the dataframe is done before passing
it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.
"""
type DataHandler{T} <: AbstractDH{T}

    # dataframe where all data is kept
    df::DataFrame

    colsInput::Array{Symbol, 1}
    colsOutput::Array{Symbol, 1}

    colsNormalize::Array{Symbol, 1}

    mu::Array{T, 1}
    norm::Array{T, 1}
    userange::Bool

    # training data, is a subset of df
    dfTrain::DataFrame
    # testing data, is a subset of df
    dfTest::DataFrame

    # these are the arrays of training, testing data
    X_train::Array{T}
    y_train::Array{T}
    X_test::Array{T}
    y_test::Array{T}

    # this is where predictions are kept
    yhat::Array{T}
    yhat_train::Array{T}

    """
    This constructor makes a fractional train, test split.  
    """
    function DataHandler(df::DataFrame; testfrac::AbstractFloat=0., shuffle::Bool=false,
                         input_cols::Array{Symbol}=Symbol[], 
                         output_cols::Array{Symbol}=Symbol[],
                         normalize_cols::Array{Symbol}=Symbol[],
                         assign::Bool=false,
                         userange::Bool=false)
        ndf = copy(df)
        o = new(ndf, input_cols, output_cols, normalize_cols)
        o.userange = userange
        split!(o, testfrac, shuffle=shuffle, assign=assign)
        computeNormalizeParameters!(o, dataset=:dfTrain)
        if canNormalize(o)
            normalizeTrain!(o)
            if size(o.dfTest)[1] > 0 normalizeTest!(o) end
        end
        return o
    end
end
export DataHandler


"""
Gets the parameters for centering and rescaling.

Does this using the training dataframe by default, but can be set to use test.
Exits normally if this doesn't need to be done for any columns.

This should always be called before normalize!, that way you have control over what
dataset the parameters are computed from.
"""
function computeNormalizeParameters!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
    if length(dh.colsNormalize)==0 return end
    df = getfield(dh, dataset)
    mu = Array{T, 1}(length(dh.colsNormalize))
    norm = Array{T, 1}(length(dh.colsNormalize))
    for (i, col) in enumerate(dh.colsNormalize)
        mu[i] = mean(df[col])
        if dh.userange
            norm[i] = maximum(df[col]) - minimum(df[col])
        else
            norm[i] = std(df[col])
        end
    end
    dh.mu = mu
    dh.norm = norm
    return
end
export computeNormalizeParameters!


"""
Determines whether the data in the datahandler can be normlized, i.e. because
the parameters have or haven't been computed yet.
"""
function canNormalize(dh::AbstractDH)
    can = isdefined(dh, :mu) && isdefined(dh, :norm)
    can = can && length(dh.mu) == length(dh.colsNormalize)
    can = can && length(dh.norm) == length(dh.colsNormalize)
    return can
end


"""
Centers and rescales the appropriate columns.
"""
function normalize!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
    if !canNormalize(dh)
        error("Trying to normalize before parameters have been computed.")
    end
    df = getfield(dh, dataset)
    for (i, col) in enumerate(dh.colsNormalize)
        df[col] = (df[col] - dh.mu[i])./dh.norm[i]
    end
    return
end
export normalize!
normalizeTrain!(dh::AbstractDH) = normalize!(dh, dataset=:dfTrain)
export normalizeTrain!
normalizeTest!(dh::AbstractDH) = normalize!(dh, dataset=:dfTest)
export normalizeTest!


"""
Performs the inverse of the centering and rescaling operations on an array.
"""
function unnormalize!{T}(dh::AbstractDH{T}, X::Array{T, 2}, cols::Array{Symbol, 1})
    err = "Array must have same number of dimensions as are to be inverted."
    @assert size(X)[2] == length(cols) err
    for (i, col) in enumerate(cols)
        idx = find(dh.colsNormalize .== col)[1]
        X[:, i] = dh.norm[idx]*X[:, i] + dh.mu[idx]
    end
end
export unnormalize!


"""
Performs the inverse of the centering and rescaling operations on a rank-1 array.
"""
function unnormalize!{T}(dh::AbstractDH{T}, X::Array{T, 2}, col::Symbol)
    unnormalize!(dh, X, [col])
end
export unnormalize!


"""
Shuffles the main dataframe of the DataHandler.
"""
function shuffle!(dh::AbstractDH)
    shuffle!(dh.df)
end
export shuffle!


"""
Assigns the training data in the data handler.

Note that this is silent if the training dataframe is empty.
"""
function assignTrain!{T}(dh::AbstractDH{T})
    if isempty(dh.dfTrain) return end
    X = convert(Array{T}, dh.dfTrain[:, dh.colsInput])
    y = convert(Array{T}, dh.dfTrain[:, dh.colsOutput])
    dh.X_train = X
    dh.y_train = y
    return X, y
end
export assignTrain!


"""
Assigns the test data in the data handler.

Note that this is silent if the test dataframe is empty.
"""
function assignTest!{T}(dh::AbstractDH{T})
    if isempty(dh.dfTest) return end
    X = convert(Array{T}, dh.dfTest[:, dh.colsInput])
    y = convert(Array{T}, dh.dfTest[:, dh.colsOutput])
    dh.X_test = X
    dh.y_test = y
    return X, y
end
export assignTest!


"""
Assigns training and test data in the data handler.
"""
function assign!{T}(dh::AbstractDH{T})
    assignTrain!(dh)
    assignTest!(dh)
    return
end
export assign!


"""
Gets the training data input, output tuple.

If flatten, attempts to flatten y.
"""
function getTrainData(dh::AbstractDH; flatten::Bool=false)
    X, y = dh.X_train, dh.y_train
    if flatten
        @assert size(y)[2] == 1 "Attempted to flatten rank-2 array."
        y = squeeze(y, 2)
    end
    return X, y    
end
export getTrainData


"""
Gets the test data input, output tuple.
"""
function getTestData(dh::AbstractDH; flatten::Bool=false)
    X, y = dh.X_test, dh.y_test
    if flatten
        @assert size(y)[2] == 1 "Attempted to flatten rank-2 array."
        y = squeeze(y, 2)
    end
    return X, y
end
export getTestData


"""
Creates a train, test split by index.  The index given is the last index of the training set.
"""
function split!(dh::AbstractDH, index::Integer; assign::Bool=true)
    @assert index ≤ size(dh.df)[1] "Index value too large for dataframe."
    dh.dfTrain = dh.df[1:index, :]
    if index == size(dh.df)[1]
        dh.dfTest = DataFrame()
    else
        dh.dfTest = dh.df[(index+1):end, :]
    end
    if assign assign!(dh) end
    return
end
export split!


"""
Creates a train, test split by fraction.  The fraction given is the test fraction.
"""
function split!(dh::AbstractDH, testfrac::AbstractFloat; shuffle::Bool=false,
                assign::Bool=true)
    @assert testfrac ≤ 1.0 "Test fraction must be ∈ [0, 1]."
    if shuffle shuffle!(dh) end
    index = convert(Int64, round((1. - testfrac)*size(dh.df)[1]))
    split!(dh, index, assign=assign)
    return 
end
export split!


#========================================================================================
NOTE to self:
    I was getting seriously confused about the proper use of eval, especially as it
    related to macros.  Because it always runs in global scope, I thought it had
    to almost always be avoided.

    The key thing that I was missing is that it is possible to pass it literal values,
    as opposed to just symbols.  By passing a literal value to the expression in eval,
    it's ok to evaluate objects the come from local scope in global scope.
========================================================================================#
# TODO make functional version of split using getfield.


"""
Used by constrain and split macros.  Looks through expressions for symbols of 
columns, and replaces them with proper ref calls.
"""
function _replaceExprColNames!(expr::Expr, dh::AbstractDH, dfname::Symbol)
    for (i, arg) in enumerate(expr.args)
        # check for nested expressions
        if isa(arg, Expr) 
            _replaceExprColNames!(expr.args[i], dh, dfname)
        elseif isa(arg, Symbol) && (arg ∈ names(getfield(dh, dfname)))
            argsymb = Expr(:quote, arg)
            expr.args[i] = :($dh.$dfname[$argsymb]) 
        end
    end
    return expr
end


"""
Constrains the training dataframe to satisfy the provided constraint.

Input should be in the form of ColumnName relation value.  For example
x .> 3 imposes df[:x] .> 3.
"""
macro constrainTrain!(dh, constraint)
    constr_expr = Expr(:quote, constraint)
    o = quote
        local constr = DatasToolbox._replaceExprColNames!($constr_expr, $dh, :dfTrain)
        constr = eval(constr)
        $dh.dfTrain = $dh.dfTrain[constr, :]
    end
    return esc(o)
end
export @constrainTrain!


"""
Constrains the test dataframe to satisfy the provided constriant.

Input should be in the form of ColumnName relation value.  For example,
x .> 3 imposes df[:x] .> 3.
"""
macro constrainTest!(dh, constraint)
    constr_expr = Expr(:quote, constraint)
    o = quote
        local constr = DatasToolbox._replaceExprColNames!($constr_expr, $dh, :dfTest)
        constr = eval(constr)
        $dh.dfTest = $dh.dfTest[constr, :]
    end
    return esc(o)
end
export @constrainTest!


"""
Constrains either the test or training dataframe.  See the documentation for those.

Note that these constraints are applied directly to the train or test dataframe,
so some of the columns may be transformed in some way.

For example: `@constrain dh test x .≥ 3`
"""
macro constrain!(dh, traintest, constraint)
    constr_expr = Expr(:quote, constraint)
    if lowercase(string(traintest)) == "train"
        dfname = Expr(:quote, :dfTrain)
    elseif lowercase(string(traintest)) == "test"
        dfname = Expr(:quote, :dfTest)
    else
        throw(ArgumentError("Dataframe to constrain must be either train or test."))
    end
    o = quote
        local constr = DatasToolbox._replaceExprColNames!($constr_expr, $dh, $dfname)
        constr = eval(constr)
        setfield!($dh, $dfname, getfield($dh, $dfname)[constr, :])
    end
    return esc(o)
end
export @constrain!


"""
Used by @split, for internal use only.
"""
function _apply_constraint!(dh::AbstractDH, constraint::BitArray)
    dh.dfTest = dh.df[constraint, :]
    dh.dfTrain = dh.df[!constraint, :]
end


"""
Splits the data into training and test sets.  The test set will be the data for which
the provided constraint is true.

For example, `@split dh x .≥ 3.0` will set the test set to be df[:x] .≥ 3.0 and the
training set to be df[:x] .< 3.0.
"""
macro split!(dh, constraint)
    constr_expr = Expr(:quote, constraint)
    o = quote
        local constr = DatasToolbox._replaceExprColNames!($constr_expr, $dh, :df)
        constr = eval(constr)
        DatasToolbox._apply_constraint!($dh, constr)
    end
    return esc(o)
end
export @split!


"""
Creates a dataframe from the test dataframe and a supplied prediction.  

For the time being this only works if the prediction is a vector.
"""
function getTestAnalysisData(dh::AbstractDH, ŷ::Array; name::Symbol=:ŷ)
    @assert length(size(ŷ)) == 1 ("getTestAnalysisData hasn't been implemented
                                  for higher rank predictions yet.")
    df = copy(dh.dfTest)
    df[name] = ŷ
    return df
end
export getTestAnalysisData




