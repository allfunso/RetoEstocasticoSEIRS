using Plots
using Random
using JLD2

#=
Este código se emplea para generar una base de datos que sirva para entrenar una red neuronal
Por lo tanto, se realiza la simulación varias veces y el vector de infectados se guarda en un archivo
Normalizar la población y poner mismas condiciones iniciales nos ahorra parte del preprocesamiento, 
pero es algo que se debe considerar en la aplicación para que el modelo sea efectivo en predicciones.
Se toman los primeros 30 días y en base a eso decide qué tan grande será el pico.
=#

prediction_span = 30

#Valores iniciales de la poblacion (normalizado)
N = 1000; #Numero total de la poblacion

# Se repite toda la simulación para obtener una gran cantidad de datos

R0 = 0/N; #Recuperados
I0 = 1/N; #Infectados
E0 = 0/N; #Expuestos
S0 = 1 - I0; #Susceptibles

#Valores iniciales para el tiempo 

D = 100; #Dias totales de simulacion
dt = 0.1; #Delta del tiempo tomado
t = 0:dt:D;
span = length(t); #Numero de pasos para la simulacion

shots = 100
I_data = zeros(shots, prediction_span+1)

for shot = 1:shots


    # Probabilidades iniciales y constantes beta, sigma, gamma y mu
    beta = 1; # Tasa de transmision
    sigma = 1/7; # Tiempo de incubacion
    gamma = 1/7; # Tasa de recuperacion
    mu = 1/120 # Tasa de pérdida de inmunidad

    #Generar las matrices de las poblaciones
    S = zeros(span);
    E = zeros(span);
    I = zeros(span);
    R = zeros(span);

    S[1] = S0;
    E[1] = E0;
    I[1] = I0;
    R[1] = R0;

    ################

    # Generar el proceso de markov

    for i = 2:span

        # Probabilidades en funcion de la cantidad de personas por
        # categoria (2do caso) considerando poblacion normalizada
        pe = dt*beta*S[i-1]*I[i-1];
        pi = dt*sigma*E[i-1];
        pr = dt*gamma*I[i-1];
        ps = dt*mu*R[i-1];

        # Generar numeros aleatorios
        Rns = rand(Int(round(S[i-1]*N)));
        Rne = rand(Int(round(E[i-1]*N)));
        Rni = rand(Int(round(I[i-1]*N)));
        Rnr = rand(Int(round(I[i-1]*N)));

        # Conseguir el numero de personas que cumplen con 
        # las probabilidades establecidas (normalizado)
        dS = sum(pe .> Rns)/N;
        dE = sum(pi .> Rne)/N;
        dI = sum(pr .> Rni)/N;
        dR = sum(ps .> Rnr)/N;

        #Obtener los nuevos valores de la poblacion
        S[i] = S[i-1] - dS + dR;
        E[i] = E[i-1] + dS - dE;
        I[i] = I[i-1] + dE - dI;
        R[i] = R[i-1] + dI - dR;
    end

end

# Tiempo final en la grafica
tf = Int(round(span / 2));

# Graficacion
plot(t[1:tf],S[1:tf]*100, label = "Poblacion susceptible", legend=:right)
plot!(t[1:tf],E[1:tf]*100, label = "Poblacion expuesta")
plot!(t[1:tf],I[1:tf]*100, label = "Poblacion infectada")
plot!(t[1:tf],R[1:tf]*100, label = "Poblacion Recuperada")

#Propiedades de la grafica
title!("Modelo SEIRS (Poblacion normalizada)")

xlabel!("Tiempo (dias)")
ylabel!("% de la Poblacion Total")

# Guardar la matriz para datos de entrenamiento
save_object("Infec_dataset.jld2", I_data)