using Flux
using Flux: @epochs
using Base.Iterators: repeated
using JLD2

I_data = load_object("Infec_dataset.jld2")

# Define the architecture of the neural network
model = Chain(
    # Reshape the input to (1, 100, 1) where (channels, height, width)
    x -> reshape(x, 1, 100, 1),
    Conv((3, 1) => 16, relu),
    MaxPool((2, 1)),
    Conv((3, 1) => 32, relu),
    MaxPool((2, 1)),
    x -> reshape(x, :, size(x, 4)),  # Flatten for fully connected layers
    Dense(32, 10, relu),
    Dense(10, 1)  # Output layer with 1 neuron
)

# Generate a random input vector of 100 elements
input_vector = rand(100)

# Pass the input through the model to get the output
output = model(input_vector)
