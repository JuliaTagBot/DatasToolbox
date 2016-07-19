
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
                         output_cols::Array{Symbol}=Symbol[])
        ndf = copy(df)
        o = new(ndf, input_cols, output_cols)
        split!(o, testfrac, shuffle=shuffle)
        return o
    end
end
export DataHandler


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
    return
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
    return
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
"""
function getTrainData(dh::AbstractDH)
    return dh.X_train, dh.y_train
end
export getTrainData


"""
Gets the test data input, output tuple.
"""
function getTestData(dh::AbstractDH)
    return dh.X_test, dh.y_test
end
export getTestData


"""
Creates a train, test split by index.  The index given is the last index of the training set.
"""
function split!(dh::AbstractDH, index::Integer; assign::Bool=true)
    @assert index <= size(dh.df)[1]
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
    @assert testfrac <= 1.0
    if shuffle shuffle!(dh) end
    index = convert(Int64, round((1. - testfrac)*size(dh.df)[1]))
    split!(dh, index, assign=assign)
    return 
end
export split!


"""
Used by @split, for internal use only.

Generates useable constraints given those given algabreically.
"""
function _get_constraint(dhsymb::Symbol, names::Array{Symbol, 1}, compare::Expr)
    args = copy(compare.args)
    for i in (1, 3)
        if args[i] in names
            argsymb = Expr(:quote, args[i])
            args[i] = Expr(:ref, :($dhsymb.df), argsymb)
        end
    end
    return Expr(:comparison, args[1], args[2], args[3])
end


"""
Used by @split, for internal use only.
"""
function _apply_constraint!(dh::AbstractDH, constraint::BitArray)
    dh.dfTest = dh.df[constraint, :]
    dh.dfTrain = dh.df[!constraint, :]
end


"""
Splits the dataframe into training and test sets using nice syntax.

For example, can do `@split dh x >= 3`.  This will make the test set of datapoints for
which dh.df[:x] >= 3.
"""
macro split(dh, constraint)
    dhsymb = Expr(:quote, dh)
    constrsymb = Expr(:quote, constraint)
    # for now we separate into different types of expressions, starting with comparison
    if constraint.head == :comparison
        o = quote
            local names_ = DataFrames.names($dh.df)
            local constr = DatasToolbox._get_constraint($dhsymb, names_, $constrsymb)
            local constraint = eval(constr)
            DatasToolbox._apply_constraint!($dh, constraint)
        end
    end
    return esc(o)
end
export @split


