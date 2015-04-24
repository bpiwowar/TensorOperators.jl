
# --- Sequence

type SequenceOperator
    list::Array{Operator}

    SequenceOperator() = new([])
end

function push!(sequence::SequenceOperator, a_module::Operator)
    push!(sequence.list, a_module)
end

function forward(sequence::SequenceOperator, input)
    for a_module = sequence.list
        forward(a_module, input)
    end
end
