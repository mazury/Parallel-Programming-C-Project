#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

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

void createShannoGradient(double x[], double g[], int dim, int procs, int myId) {
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

void createQuarticGradient(double x[], double subg[], int dim, int procs, int myId) {
        int i;

        for (i = myId *dim / procs; i < dim / procs * (1 + myId); i++) {
           subg[i] = 2 * sqrt(quartic(x, dim)) * 2 * (i + 1)  * x[i];
        }
}

int find_minimum_index(int a[], int n) {
  int c, min, index;
 
  min = a[1];
  index = 1;
 
  for (c = 2; c < n; c++) {
    if (a[c] < min) {
       index = c;
       min = a[c];
    }
  }
 
  return index;
}

int find_minimum(int a[], int n) {
  int c, min, index;
 
  min = a[1];
  index = 1;
 
  for (c = 2; c < n; c++) {
    if (a[c] < min) {
       index = c;
       min = a[c];
    }
  }
 
  return min;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int numprocs, myId;
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &myId);


    int dim = atoi(argv[2]);
    double *x = (double *) malloc(dim * sizeof(double));
    double *d = (double *) malloc(dim * sizeof(double));
    double *g = (double *) malloc(dim * sizeof(double));
    double *subg = (double *) malloc(dim * sizeof(double));
    double *newG = (double *) malloc(dim * sizeof(double));
    double *testX = (double *) malloc(dim * sizeof(double));
    int *workersAlfaIter = (int *) malloc(dim * sizeof(int));
    double *workersAlfa = (double *) malloc(dim * sizeof(double));
  
    double (*func)(double x[], int dim);
    void (*createGradient)(double x[], double g[], int dim, int procs, int myId);
    double initialValue;
    if (strcmp(argv[1], "quartic") == 0) {
        func = quartic;
        createGradient = createQuarticGradient;
        initialValue = 1;
    } else if (strcmp(argv[1], "shanno") == 0) {
        func = shanno;
        createGradient = createShannoGradient;
        initialValue = 0;
    } 
    int minimumFound = 0;
    double alfa, betaNumerator, betaDenominator, norm, beta, oldFunc, b, workerAlfa, deltab;
    double accuracy = 1e-7;
    double gamma = 0.95, delta = 0.1;
    int i, workerAlfaIter, alfaIter, alfaSet, iter = 1;
    double totalAlfaTime, totalTime, totalGradientTime;
    double startAlfaTime, startTime, startGradientTime;
    int test = 0;
    MPI_Status status;
    MPI_Request req;
  
    // wypełnienie wektorów x, g oraz d wartościami początkowymi - punkt startowy algorytmu
    for (i = 0; i < dim; i++)
        x[i] = initialValue;

    if (func == quartic) {
        createGradient(x, subg, dim, numprocs, myId);
        MPI_Reduce(subg, g, dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        createGradient(x, g, dim, numprocs, myId);
    }

    for (i = 0; i < dim; i++)
        d[i] = g[i];
  
    // główna pętla
    while(minimumFound == 0) {
        if (myId == 0) {
            startTime = MPI_Wtime();
            b = 0;

            for (i = 0; i < dim; i++) {
                b += d[i] * g[i];
            }
        }

        // wysłanie pozostałym procesom zmiennych potrzbnych do obliczenia kroku
        if (func == shanno) {
            MPI_Bcast(x, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(d, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        alfaIter = myId;
        alfaSet = 0;
        oldFunc = func(x, dim);
        deltab = b * delta;

        // nieblokujące odebranie wartości kroku oraz numerów iteracji, w których workerzy znaleźli krok
        if (func == shanno && numprocs != 1 && myId == 0) {
            for (i = 1; i < numprocs; i++) {
                MPI_Irecv(&workersAlfa[i], 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, &req);
                MPI_Irecv(&workersAlfaIter[i], 1, MPI_INT, i, 2, MPI_COMM_WORLD, &req);
            }
        }

        // obliczanie długości kroku
        if (func == shanno || (func == quartic && myId == 0)) {
            startAlfaTime = MPI_Wtime();

            while (alfaSet == 0) {
                alfaIter = (func == quartic) ? alfaIter + 1 : alfaIter + numprocs;
                alfa = (func == quartic) ? pow(gamma, alfaIter - 1) : pow(gamma, alfaIter - numprocs);
                
                for (i = 0; i < dim; i++)
                    testX[i] = x[i] + alfa * d[i];

                // sprawdzenie warunku czy aktualny krok jest właściwy
                if (func(testX, dim) - alfa * deltab <= oldFunc) {

                    // wysłanie przez workerów kroku oraz iteracji, w której znaleźli krok do mastera
                    if (func == shanno && myId != 0) {
                        alfaSet = 1;
                        MPI_Send(&alfa, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
                        MPI_Send(&alfaIter, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                    }

                    // synchronizacja procesów
                    if (func == shanno)
                        MPI_Barrier(MPI_COMM_WORLD);

                    // ustalenie właściwego kroku przez mastera
                    if (myId == 0) {
                        alfaSet = 1;
                        if (func == shanno && numprocs != 1) {
                            int minWorkersIter = find_minimum(workersAlfaIter, numprocs);
                            int minWorkersIterIndex = find_minimum_index(workersAlfaIter, numprocs);
                            if (alfaIter > minWorkersIter)
                                alfa = workersAlfa[minWorkersIterIndex];
                        }
                    }
                }
            }                 
            totalAlfaTime += MPI_Wtime() - startAlfaTime;
        }

        if (myId == 0) {
            betaNumerator = 0;
            betaDenominator = 0;
            norm = 0;

            for (i = 0; i < dim; i++)
                x[i] += alfa * d[i];
        }

        startGradientTime =  MPI_Wtime();

        // zrównoleglenie tworzenia wektora gradientu
        if (func == quartic) {
            MPI_Bcast(x, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            createGradient(x, subg, dim, numprocs, myId);
            MPI_Reduce(subg, newG, dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

    
        if (myId == 0) {
            if (func == shanno)
                createGradient(x, newG, dim, numprocs, myId);
            
            totalGradientTime += MPI_Wtime() - startGradientTime;
            
            // obliczanie wektora d oraz normy w kolejnej iteracji
            for (i = 0; i < dim; i++) {
                betaNumerator += newG[i] * (newG[i] - g[i]);
                betaDenominator += pow(g[i], 2);
                d[i] = -1 * newG[i] + beta * d[i];
                norm += pow(newG[i], 2);
            }

            beta = betaNumerator / betaDenominator;
            norm = sqrt(norm);

            // zbiegnięcie
            if (norm < accuracy) {
                minimumFound = 1;

                // FILE *fp;
                // fp=freopen("OUT", "a" ,stdout);
                printf("%d, %.3lf, %.3lf, %.3lf\n", dim, totalTime, totalAlfaTime / totalTime, totalGradientTime / totalTime);
                // fclose(fp);
            } 

            for (i = 0; i < dim; i++)
                g[i] = newG[i];
        }

        iter++;
        totalTime +=  MPI_Wtime() - startTime;
    }

    // printf("Całkowity czas: %.3lf\n", totalTime);
    // printf("Udział czasu części liczącej długość kroku: %.3lf\n", totalAlfaTime / totalTime);
    // printf("Udział czasu części liczącej gradient: %.3lf\n", totalGradientTime / totalTime);
    // printf("Liczba iteracji: %d\n", iter);
    // printf("Minimum: x0 = %f\n", x[0]);

    MPI_Abort(MPI_COMM_WORLD, 0);
    MPI_Finalize();
    return 0;
}
