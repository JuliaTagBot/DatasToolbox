using DatasToolbox
using DatasToolbox.AutoSer

const dir = "autoser_test"

function f(a, ϕ)
    println("being called")
    a*e^(im*ϕ)
end


# y = autoser(dir, f, :f, 1.0, π)

mac = macroexpand(:(@autoser dir y = f(1.0, π)))


@autoser dir y = f(1.0, π)

