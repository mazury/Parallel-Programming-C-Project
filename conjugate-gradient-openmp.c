#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

double shanno(double x[], int dim) {
    double sum = pow(2 * x[0] - 1, 2);
    int i;
    for (i = 0; i < dim - 1; i++) {
        sum += ((i + 2) * pow(2 * x[i] - x[i + 1], 2));
    }
    return sum;
}

double quartic(double x[], int dim) {
    int i;
    double sum = 0;
    for (i = 0; i < dim; i++) {
        sum += (double)(i + 1) * pow(x[i], 2);
    }
    return pow(sum, 2);
}

void createShannoGradient(double x[], double g[], int dim, int threads) {
    int i;
    for (i = 0; i < dim; i++) {
        if (i == 0) {
            g[i] = 24 * x[0] - 8 * x[1] - 4; 
        } else if (i < dim - 1) {
            g[i] = (-4 * i - 4) * x[i-1] + (10 * (i + 1) + 8) * x[i] - (i + 2) * 4 * x[i +1];
        } else {
            g[i] = (-4 * i - 4) * x[i - 1] + (i * 2 + 2) * x[i]; 
        }
    }
}

void createQuarticGradient(double x[], double g[], int dim, int threads) {
    #pragma omp parallel num_threads(threads)
    {
        int i;
        #pragma omp for
        for (i = 0; i < dim; i++) {
            g[i] = 2 * sqrt(quartic(x, dim)) * 2 * (i + 1)  * x[i];
        }
    }
}

int main(int argc, char *argv[]) {
    // informacje dla użytkownika programu
    if (argc > 8 || argc < 7) {
        printf("\nParametry programu:\n \
            1. testowana funkcja (shanno lub quartic)\n \
            2. liczba wymiarów\n \
            3. dokładność (np. 1e-6 albo 0.000001\n \
            4. wsp. gamma (0 - 1)\n \
            5. liczba wątków wykorzystanych przy obliczaniu kroku\n \
            6. liczba wątków wykorzystanych przy obliczaniu gradientu\n\n");
    }

    // deklaracja i inicjalizacja zmiennych i funkcji
    int dim = atoi(argv[2]);
    double accuracy = atof(argv[3]);
    double *x = (double *) malloc(dim * sizeof(double));
    double *d = (double *) malloc(dim * sizeof(double));
    double *g = (double *) malloc(dim * sizeof(double));
    double *newG = (double *) malloc(dim * sizeof(double));
    double (*func)(double x[], int dim);
    void (*createGradient)(double x[], double g[], int dim, int threads);
    double initialValue;
    if (strcmp(argv[1], "quartic") == 0) {
        func = quartic;
        createGradient = createQuarticGradient;
        initialValue = 1;
    } else if (strcmp(argv[1], "shanno") == 0) {
        func = shanno;
        createGradient = createShannoGradient;
        initialValue = -1;
    }
    bool minimumFound = false;
    double alfa, betaNumerator, betaDenominator, norm, beta;
    double gamma = atof(argv[4]);
    double delta = 0.3;
    int iter = 1;
    double alfaStart, totalAlfaTime, gradientCreatingStart, totalGradientCreatingTime;
    int alfaThreads = atoi(argv[5]);
    int gradientThreads = atoi(argv[6]);
    int i;

    // wypełnienie wektorów x, g oraz d wartościami początkowymi - punkt startowy algorytmu
    for (i = 0; i < dim; i++) {
        x[i] = initialValue;
    }
    createGradient(x, g, dim, gradientThreads);
    for (i = 0; i < dim; i++) {
        d[i] = g[i];
    }

    // uruchomienie liczników czasu
    double start = omp_get_wtime();
    
    // główna pętla
    while(!minimumFound) {
        double b = 0;

        for (i = 0; i < dim; i++) {
            b += d[i] * g[i];
        }

        double oldFunc = func(x, dim);
        alfaStart = omp_get_wtime();
        bool alfaSet = false;

        // zrównoleglony obszar krytyczny poszukujący długości kroku
        #pragma omp parallel num_threads(alfaThreads)
        {
        int alfaIter = omp_get_thread_num();
        int step = omp_get_num_threads();
        double localAlfa = alfa;
        double *testX = (double *) malloc(dim * sizeof(double));
        while (!alfaSet) {
            alfaIter += step;
            localAlfa = pow(gamma, alfaIter - 1);
            for (i = 0; i < dim; i++) {
                testX[i] = x[i] + localAlfa * d[i];
            }
            if (func(testX, dim) - delta * localAlfa * b <= oldFunc) {
                #pragma omp critical
                {
                    if (!alfaSet) {
                        alfaSet = true;
                        alfaIter = 0;
                        alfa = localAlfa;
                    }
                }
            } 
        }
        free(testX);
        }

        totalAlfaTime += omp_get_wtime() - alfaStart;

        betaNumerator = 0;
        betaDenominator = 0;
        norm = 0;


        // obliczanie wektorów x, g i d oraz normy w kolejnej iteracji
        for (i = 0; i < dim; i++) {
            x[i] += alfa * d[i];
        }

        gradientCreatingStart = omp_get_wtime();
        createGradient(x, newG, dim, gradientThreads);
        totalGradientCreatingTime += omp_get_wtime() - gradientCreatingStart;
        
        for (i = 0; i < dim; i++) {
            betaNumerator += newG[i] * (newG[i] - g[i]);
            betaDenominator += pow(g[i], 2);
            d[i] = -1 * newG[i] + beta * d[i];
            norm += pow(newG[i], 2);
        }

        beta = betaNumerator / betaDenominator;
        norm = sqrt(norm);
        

        // wyświetlanie na ekranie w celu obserwacji zbiegania
        if (argv[7]) {
            if (iter % atoi(argv[7]) == 0 ) {
                printf("x0 = %.10lf\n", x[0]);
            } 
        }


        // zbiegnięcie
        if (norm < accuracy) {
            minimumFound = true;
            double time = omp_get_wtime() - start;
            printf("\n");
            printf("Czas rozwiązania: %lf\n", time);
            printf("Udział czasu spędzonego przy obliczaniu kroku alfa: %.2lf\n", totalAlfaTime / time);
            printf("Udział czasu spędzonego przy obliczaniu gradientu: %.2lf\n", totalGradientCreatingTime / time);
            printf("Liczba iteracji: %d\n", iter);
            printf("Minimum:");
            printf("x0 = %f\n", x[0]);
            printf("\n");
        } 

        for (i = 0; i < dim; i++) {
            x[i] = x[i];
            g[i] = newG[i];
        }

        iter++;
    }
    return 0;
}