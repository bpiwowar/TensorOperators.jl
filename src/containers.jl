export Sequence, Parallel

# --- Sequence

type Sequence
    list::Array{Layer}

    Sequence() = new([])
end

function push!(s::Sequence, a_module::Layer)
    push!(s.list, a_module)
end

function forward!(s::Sequence, input)
    for a_module = s.list
        forward!(a_module, input)
    end
end


# --- Parallel

type Parallel
    list::Array{Layer}

    Parallel() = new([])
end

function push!(p::Parallel, a_module::Layer)
    push!(p.list, a_module)
end

function forward!(p::Parallel, input::Array)
    @assert length(input) == length(p.list)
    for a_module = p.list
        forward!(a_module, input)
    end
end


ListBasedContainer = Union{Parallel, Sequence}

function init!(m::ListBasedContainer)
    for submodule in m.list
        init!(submodule)
    end
end

function init_gradient!(m::ListBasedContainer)
    for submodule in m.list
        init_gradient!(submodule)
    end
end
