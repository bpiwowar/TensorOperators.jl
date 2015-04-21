
# --- Sequence

type SequenceModule
    list::Array{Module}

    SequenceModule() = new([])
end

function push!(sequence::SequenceModule, a_module::Module)
    push!(sequence.list, a_module)
end

function forward(sequence::SequenceModule, input)
    for a_module = sequence.list
        forward(a_module, input)
    end
end
