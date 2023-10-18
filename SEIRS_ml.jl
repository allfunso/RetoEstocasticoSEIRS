using Plots
using JLD2
using Random

#=
Este código se emplea para generar una base de datos que sirva para entrenar una red neuronal
Por lo tanto, se realiza la simulación varias veces y el vector de infectados se guarda en un archivo
Normalizar la población y poner mismas condiciones iniciales nos ahorra parte del preprocesamiento, 
pero es algo que se debe considerar en la aplicación para que el modelo sea efectivo en predicciones.
Se toman los primeros 40 días y en base a eso decide qué tan grande será el pico.
=#

prediction_span = 40 # Número de días que se proporciona al modelo

# Datos iniciales de la poblacion

N = 10000; # Número total de la población
R0 = 0; # Recuperados
I0 = 10; # Infectados
E0 = 0; # Expuestos
S0 = N - I0; # Susceptibles

# Tiempo de simulación

tf = 200; # Días totales de simulacion
dt = 1; # Delta del tiempo
t = 0:dt:tf;
iterations = length(t); #Numero de pasos para la simulacion

#######################

# Se repite toda la simulación para obtener una gran cantidad de datos

shots = 500
I_data = zeros(prediction_span + 2, shots)

for shot = 1:shots

    # Parámetros beta, sigma, gamma y mu

    beta = randn()*0.3^2 + 0.3; # Tasa de transmisión
    sigma = randn()*0.4^2 + 0.4; # Tiempo de incubación
    gamma = randn()*0.1^2 + 0.1; # Tasa de recuperación
    mu = 1/365; # Tasa de pérdida de inmunidad

    # Generar los vectores de las poblaciones
    S = zeros(iterations);
    E = zeros(iterations);
    I = zeros(iterations);
    R = zeros(iterations);

    S[1] = S0;
    E[1] = E0;
    I[1] = I0;
    R[1] = R0;

    ################

    # Generar el proceso de markov

    for i = 2:iterations
        # Probabilidades
        pSE = dt*beta * I[i-1]/N;
        pEI = dt*sigma;
        pIR = dt*gamma;
        pRS = dt*mu;

        # Generar numeros aleatorios
        RnS = rand(Int(S[i-1]));
        RnE = rand(Int(E[i-1]));
        RnI = rand(Int(I[i-1]));
        RnR = rand(Int(R[i-1]));

        # Conseguir el numero de personas que cumplen con las probabilidades establecidas
        dS = sum(pSE .> RnS);
        dE = sum(pEI .> RnE);
        dI = sum(pIR .> RnI);
        dR = sum(pRS .> RnR);

        #Obtener los nuevos valores de la poblacion
        S[i] = S[i-1] - dS + dR;
        E[i] = E[i-1] - dE + dS;
        I[i] = I[i-1] - dI + dE;
        R[i] = R[i-1] - dR + dI;
    end
    # Extracción de datos y preprocesamiento (normalización)
    start_day = Int(round(rand()*14) + 1);
    end_day = start_day + prediction_span - 1;
    I_norm = I ./ N;
    I_max, idx = findmax(I_norm);
    days_to_max = idx - end_day;
    I_data[:, shot] = vcat(I_norm[start_day:end_day], [I_max, days_to_max]);
end

# Guardar la matriz para datos de entrenamiento y prueba
save_object("Infec_dataset.jld2", I_data)