
# --- Sequence

type SequenceContainer
    list::Array{Operator}

    SequenceContainer() = new([])
end

function push!(sequence::SequenceContainer, a_module::Operator)
    push!(sequence.list, a_module)
end

function forward(sequence::SequenceContainer, input)
    for a_module = sequence.list
        forward(a_module, input)
    end
end


# --- Parallel

type ParallelContainer
    list::Array{Operator}

    ParallelContainer() = new([])
end

function push!(sequence::ParallelContainer, a_module::Operator)
    push!(sequence.list, a_module)
end

function forward(sequence::ParallelContainer, input::Any[])
    @assert length(input) ==
    for a_module = sequence.list
        forward(a_module, input)
    end
end
