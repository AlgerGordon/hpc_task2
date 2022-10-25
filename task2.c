//
// Created by general on 23.10.22.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    // Variant 6:
    //  SSS (sin(x^2 + z^2) * y) dx dy dz, over region G = {(x, y, z): x^2 + y^2 +z^2 <= 1, x >=0, y>=0, z>=0 }
    // For Monte-Carlo we will use region P = {(x, y, z):  0 <= x <= 1, 0 <= y <= 1, 0 <= z <= 1}

    const double pi = 4.0 * atan(1.0);
    double exact_integral_value = pi / 8.0 * (1 - sin(1.0));    // 0.06225419868
    double P_region_volume = 1.0 * 1.0 * 1.0;

    double integral_approx;
    double x_rand, y_rand, z_rand;
    double func_sum;
    int n_dots_per_iter = 100;
    int n_dots = 0;
    double cur_iteration_sum;
    int iterations = 0;
    double total_sum = 0.0;             // integral_approx = P_region_volume * (1.0 / iterations) * total_sum
    double error;                       // abs(integral_approx - exact_integral_value)

    double eps;
    double max_duration;

    int num_procs, my_rank, root = 0;

    if (argc > 1 && argv[1] != NULL) {
        sscanf(argv[1], "%lf", &eps);
    } else {
        fprintf(stderr,"eps is missing!\n");
        return 1;
    }

    int seed_bias = 0;
    if (argc > 2 && argv[2] != NULL) {
        sscanf(argv[2], "%d", &seed_bias);
    } else {
        fprintf(stderr,"seed_bias is missing!\n");
        return 2;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int my_seed = seed_bias * 100 + my_rank;
    srand(my_seed);

    double start = MPI_Wtime();

    do {
        iterations += 1;
        func_sum = 0;
        for (size_t i = 0; i < n_dots_per_iter; ++i) {
            x_rand = (double) rand() / RAND_MAX;
            y_rand = (double) rand() / RAND_MAX;
            z_rand = (double) rand() / RAND_MAX;
            if ((x_rand * x_rand + y_rand * y_rand + z_rand * z_rand) > 1) {
                func_sum += 0;
            } else {
                func_sum += sin(x_rand * x_rand + z_rand * z_rand) * y_rand;
            }
        }

        MPI_Reduce(&func_sum, &cur_iteration_sum, 1,
                   MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

        if (my_rank == root) {
            n_dots += n_dots_per_iter;
            total_sum += (1.0 / num_procs) * cur_iteration_sum;
            integral_approx = P_region_volume * (1.0 / n_dots) * total_sum;
            error = fabs(integral_approx - exact_integral_value);
        }
        MPI_Barrier(MPI_COMM_WORLD);    // to make Bcast() adn Reduce kinda "synchronous"
        MPI_Bcast(&error, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
    } while (error > eps);


    double cur_rank_duration = MPI_Wtime() - start;

    MPI_Reduce(&cur_rank_duration, &max_duration, 1,
               MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
    MPI_Finalize();
    if (my_rank == root) {
        printf("num_procs: %d, eps: %.10f, time: %.10f sec,"
               " integral approx value: %.10f, error: %.10f ,"
               " dots generated: %d, seed_bias: %d\n",
               num_procs, eps, max_duration, integral_approx,
               error, num_procs * n_dots, seed_bias);
    }
    return 0;
}