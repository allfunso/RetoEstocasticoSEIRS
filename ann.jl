using JLD2
using Flux
using Flux: train!
using LinearAlgebra
using Statistics
using Plots

I_data = Float32.(load_object("Infec_dataset.jld2"))

x_train = Flux.flatten(I_data[1:40, 1:250])
y_train = (I_data[41:42, 1:250])
x_test = Flux.flatten(I_data[1:40, 251:500])
y_test = I_data[41:42, 251:500]

#= Define a simple convolutional neural network
model = Chain(
    Conv((3, 1), 40=>64, relu),
    x -> reshape(x, :, size(x, 4)),  # Flatten the output
    Dense(64 * 38, 128, relu),
    Dense(128, 2)  # Output layer with 2 units
) =#
model = Chain(
    Dense(40, 20, relu),
    Dense(20, 2)  # Output layer with 2 units
)

# Define a sample input
input = rand(40)

# Forward pass through the network
#output = model(input)
#println(output)

# Define loss function
loss(x, y) = Flux.mse(model(x), y)

# Track parameters
ps = Flux.params(model)

# Select optimizer
learning_rate = 1
opt = ADAM(learning_rate)

# Train model
loss_history = []
epochs = 1000

for epoch in 1:epochs
    # Train model
    train!(loss, ps, [(x_train, y_train)], opt)
    # Print report
    train_loss = loss(x_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

y_pred = model(x_test)

err_max = y_test[1, :] - y_pred[1, :]
err_days = y_test[2, :] - y_pred[2, :]

index = collect(1:250)
check_display = [index y_pred[1, :] y_pred[2, :] y_test[1, :] y_test[2, :] err_max err_days]

vscodedisplay(check_display)

# Plot loss
plot(1:epochs, loss_history, xlabel="epochs", ylabel="loss", title="Learning Curve")
