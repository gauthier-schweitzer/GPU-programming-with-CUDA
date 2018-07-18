#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**************************************************************
The code in time.h is a part of a course on cuda taught by its authors:
Lokman A. Abbas-Turki
**************************************************************/
#include "timer.h"

/**************************************************************
Common functions
**************************************************************/

// Compare function for qsort
int compare_function(const void *a,const void *b) {
    float *x = (float *) a;
		float *y = (float *) b;
    if (*x < *y) return - 1;
    else if (*x > *y) return 1;
    return 0;
}


// Generate gaussian vector using Box Muller
void gaussian_vector(float *v, float mu, float sigma, int n) {

    for (int i = 0; i<n; i++){
		    float u1 = (float)rand()/(float)(RAND_MAX);
		    float u2 = (float)rand()/(float)(RAND_MAX);
		    v[i] = sigma * (sqrtf( -2 * logf(u1)) * cosf(2 * M_PI * u2)) + mu;
	  }
}


//Function to print a small vector of floats on host
void print_vector(float *c, int m, int n) {

    for (int i=0; i<m; i++){
        printf("%f     ", c[i]);
        printf("\n");
 	  }
}

/**************************************************************
CPU functions
**************************************************************/
// Kernel for computing the square of a vector (INPLACE)
// We actually only need z ** 2 in the computations and not z
// The square norm is also computed
void square_vector(float *z, float *znorm, int n){
		for (int i = 0; i < n; i++) {
				float zi = z[i];
				float zsqri = zi * zi;
				z[i] = zsqri;
				znorm[0] += zsqri;
		}
}


// Function for computing f (the secular function of interest) at a given point x
float secfunc(float *d, float *zsqr, float rho, float x, int n) {

    float sum = 0;
    for (int i=0; i < n; i++){
        sum += zsqr[i] / (d[i] - x);
	  }

    return rho + sum;
}


// Function for computing f' (the prime derivative of the secular function of interest) at a given point x
float secfunc_prime(float *d, float *zsqr, float x, int n) {

    float sum = 0;
    for (int i=0; i < n; i++){
        int di = d[i];
		    sum += zsqr[i] / ((di - x) * (di - x));
    }

	  return sum;
}


// Device function for computing f'' (the second derivative of the secular function of interest)
float secfunc_second(float *d, float *zsqr, float x, int n){
    float sum = 0;

		for (int i = 0; i < n; i++) {
		    float di = d[i];
				sum += zsqr[i] / ((di - x) * (di - x) * (di - x));
		}

		return 2 * sum;
}


// Useful intermediary function, see equations (30) and (31) from Li's paper on page 13 and equation (42) on page 20
float discrimant_int(float a, float b, float c){

    if (a <= 0) return (a - sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a + sqrtf(a * a - 4 * b *c));
}


// Useful intermediary function, see equation (46) from Li's paper on page 21
float discrimant_ext(float a, float b, float c){

    if (a >= 0) return (a + sqrtf(a * a - 4 * b * c)) / (2 * c);
    else return (2 * b) / (a - sqrtf(a * a - 4 * b *c));
}


// h partition of the secular function, used for Initialization
float h_secfunc(float d_k, float d_kplus1, float zsqr_k, float zsqr_kplus1, float x){

    return zsqr_k / (d_k - x) + zsqr_kplus1 / (d_kplus1 - x);
}


// Initialization for interior roots (see section 4 of Li's paper - initial guesses from page 18)
float initialization_int(float *d, float *zsqr, float rho, int k, int n){

    float d_k = d[k];
    float d_kplus1 = d[k + 1];
    float delta = d_kplus1-d_k;
    float random = (float)rand()/(RAND_MAX+1.0);
    return d_k+random*delta;
}


// Initialization for the exterior root (see section 4 of Li's paper - initial guesses from page 18)
float initialization_ext(float *d, float *zsqr, float *znorm, float rho, int n){

    float d_nminus1 = d[n - 1];
    float d_n = d_nminus1 + znorm[0] / rho;
    float delta = d_n-d_nminus1;
    float random = (float)rand()/(RAND_MAX+1.0);
    return d_n+random*delta;
}


// Computation of a from the paper (page 13)
float a_gragg(float f, float fprime, float delta_k, float delta_kplus1){

    return (delta_k + delta_kplus1) * f - delta_k * delta_kplus1 * fprime;

}


// Computation of b from the paper (page 13)
float b_gragg(float f, float delta_k, float delta_kplus1){

    return delta_k * delta_kplus1 * f;
}


// Computation of c from the section Gragg of the paper (page 15)
float c_gragg(float f, float fprime, float fsecond, float delta_k, float delta_kplus1){

    return f - (delta_k + delta_kplus1) * fprime + delta_k * delta_kplus1 * fsecond / 2.0;

}


// Compute of the update for x (eta) for the interior roots (see section 3.1 - Iteration fomulas, pages 12 and 13)
float eta_int(float d_k, float d_kplus1, float f, float fprime, float fsecond, float x, int k, int n){

    float delta_k = d_k - x;
    float delta_kplus1 = d_kplus1 - x;
    float a = a_gragg(f, fprime, delta_k, delta_kplus1);
    float b = b_gragg(f, delta_k, delta_kplus1);
    float c = c_gragg(f, fprime, fsecond, delta_k, delta_kplus1);
    float eta = discrimant_int(a, b, c);
    return eta;
}

// Compute of the update of x (+eta) for the exterior root
float eta_ext(float d_nminus2, float d_nminus1, float f, float fprime, float fsecond, float x, int n){

    float delta_nminus2 = d_nminus2 - x;
    float delta_nminus1 = d_nminus1 - x;
    float a = a_gragg(f, fprime, delta_nminus2, delta_nminus1);
    float b = b_gragg(f, delta_nminus2, delta_nminus1);
    float c = c_gragg(f, fprime, fsecond, delta_nminus2, delta_nminus1);
    float eta = discrimant_ext(a, b, c);
    return eta;
}

// Iterate to find the k-th interior root
float find_root_int(float *d, float *zsqr, float rho, float x, int k, int n, int maxit, float epsilon, float *loss_CPU){

    int i = 0;
    float f = secfunc(d, zsqr, rho, x, n);;
    float d_k = d[k];
    float d_kplus1 = d[k + 1];

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc(d, zsqr, rho, x, n);
        float fprime = secfunc_prime(d, zsqr, x, n);
        float fsecond = secfunc_second(d, zsqr, x, n);
        float eta = eta_int(d_k, d_kplus1, f, fprime, fsecond, x, k, n);
        x += eta;
        i ++;
    }

    *loss_CPU += (float)(abs(f)/n);
    return x;
}


// Iterate to  find the last root (the exterior one)
float find_root_ext(float *d, float *zsqr, float rho, float x, int n, int maxit, float epsilon, float *loss_CPU){

    int i = 0;
    float d_nminus2 = d[n - 2];
    float d_nminus1 = d[n - 1];
    float f = secfunc(d, zsqr, rho, x, n);

    while ((i < maxit) && (fabsf(f) > epsilon)){
        f = secfunc(d, zsqr, rho, x, n);
        float fprime = secfunc_prime(d, zsqr, x, n);
        float fsecond = secfunc_second(d, zsqr, x, n);
        float eta = eta_ext(d_nminus2, d_nminus1, f, fprime, fsecond, x, n);
        x += eta;
        i ++;
    }
    *loss_CPU += (float)(abs(f)/n);
    return x;
}


void find_roots(float *xstar, float *x0, float *d, float *zsqr, float *znorm, float rho, int n, int maxit, float epsilon, float *loss_CPU){
    // We make sure that the loss is set to 0
    *loss_CPU =0;

		for (int i=0; i<n-1; i++){
				xstar[i] = find_root_int(d, zsqr, rho, x0[i], i, n, maxit, epsilon, loss_CPU);
		}

		xstar[n - 1] = find_root_ext(d, zsqr, rho, x0[n - 1], n, maxit, epsilon, loss_CPU);
}


void initialize_x0(float *x0, float *d, float *zsqr, float *znorm, float rho, int n){

		for (int i=0; i<n-1; i++){
				x0[i] = initialization_int(d, zsqr, rho, i, n);
		}

		x0[n - 1] = initialization_ext(d, zsqr, znorm, rho,  n);
}



int main (void) {
    /****************** Access for writing ******************/
    FILE *f = fopen("result_init.csv", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    fprintf(f, "n;iter;niter;time_GPU;time_CPU_rand;loss_GPU;loss_CPU_rand\n");

    /****************** Declaration ******************/
    // Declare vectors or floats
    float *d, *z, *zsqr, *znorm, *x0, *xstar, *loss_GPU, *loss_CPU;


    // rho parameter
    float rho = 10;


    // Size of arrow matrix chosen by the user
    int n, nlow, nhigh, step, niter;
    printf("\nLowest n to test? \n");
    scanf("%d", &nlow);
    printf("\nHighest n to test? \n");
    scanf("%d", &nhigh);
    printf("\nSize of the step? \n");
    scanf("%d", &step);
    printf("\nNumber of iterations of the same n to avoid stochastic error? \n");
    scanf("%d", &niter);
    printf("\n \n******************* CHOICE OF N ******************** \n");
    printf("We compare the chosen algorithms every %d n, for n between %d and %d \n", step, nlow, nhigh);
    printf("Each test is repeated %d times \n\n", niter);
    printf("\n \n********************** TESTS *********************** \n");


    for(n=nlow; n<=nhigh; n+=step){

      //Maximum number of iterations
      int maxit = 1e4;

      //Stopping criterion
      float epsilon = 1e-6;

      // Memory allocation for data
      d = (float*)malloc(n*sizeof(float));
      z = (float*)malloc(n*sizeof(float));
      zsqr = (float*)malloc(n*sizeof(float));

      for (int iter =0; iter<niter; iter++){
        // Memory allocation for computation
    		znorm = (float*)malloc(sizeof(float));
    		x0 = (float*)malloc(n*sizeof(float));
        xstar = (float*)malloc(n*sizeof(float));
        loss_GPU = (float*)malloc(sizeof(float));
        loss_CPU = (float*)malloc(sizeof(float));

        // Create instance of class Timer
        Timer TimG, TimC;


        //Fill the vector d with linear function of n
        for (int i=0; i < n; i++){
            d[i] = 2 * n - i;
        }

        // sort the vector in ascending order
        qsort(d, n, sizeof(float), compare_function);

        // Gaussian rank 1 perturbation
        float mu_z = 5;
        float sigma_z = 1;
        gaussian_vector(z, mu_z, sigma_z, n);
        gaussian_vector(zsqr, mu_z, sigma_z, n);


        /*************************************************************************
        ********************************* CPU ************************************
        *************************************************************************/

        // Start timer CPU
        TimC.start();

        // We first compute the square and squared norm
        square_vector(zsqr, znorm, n);

        // Initialization of x0
        initialize_x0(x0, d, zsqr, znorm, rho, n);


        /***************** Root computation ****************/
        // Find roots
        find_roots(xstar, x0, d, zsqr, znorm, rho, n, maxit, epsilon, loss_CPU);

        // End timer
        TimC.add();

          // Record the performance
        *loss_GPU=0;
        fprintf(f, "%d;%d;%d;%f;%f;%f;%f\n", n, iter, niter, (float)TimG.getsum(), (float)TimC.getsum(), *loss_GPU, *loss_CPU);

        // Free memory used for computation on CPU
        free(znorm);
        free(xstar);
        free(loss_CPU);
        free(loss_GPU);
      }

    printf("%d has been tested\n", n);
    // Free memory used to store data on CPU
    free(d);
    free(z);
    free(zsqr);
    }
    printf("\n \n");

    // We close the access to the file
    fclose(f);
}
