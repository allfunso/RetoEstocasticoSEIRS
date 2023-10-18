using Plots
using Random

# Parámetros beta, sigma, gamma y mu

beta = 0.3; # Tasa de transmisión
sigma = 0.4; # Tiempo de incubación
gamma = 0.1; # Tasa de recuperación
mu = 1/365; # Tasa de pérdida de inmunidad

# Datos iniciales de la poblacion

N = 10000; # Número total de la población

R0 = 0; # Recuperados
I0 = 10; # Infectados
E0 = 0; # Expuestos
S0 = N - I0; # Susceptibles

# Tiempo de simulación 

tf = 100#365*3; # Días totales de simulacion
dt = 1; # Delta del tiempo
t = 0:dt:tf;
iterations = length(t); #Numero de pasos para la simulacion


# Generar las matrices de las poblaciones
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

# Graficación
plot(t, S, label = "Población susceptible", legend=:right)
plot!(t, E, label = "Población expuesta")
plot!(t, I, label = "Población infectada")
plot!(t, R, label = "Población recuperada")

#Propiedades de la grafica
title!("Modelo SEIRS (Población de: " * string(N) * " personas)")

xlabel!("Tiempo (días)")
ylabel!("Población")

## Guardar el archivo de la grafica
#savefig("ModelosSEIR2.png")