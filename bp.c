#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <unistd.h>

#define MAX_POINTS 1000 // Adjust as needed
#define MAX_NEIGHBORS 20 // Adjust as needed

typedef struct {
    int neighbor;
    double distance;
} NeighborDistance;

typedef struct {
    NeighborDistance nd[MAX_POINTS][MAX_NEIGHBORS];
    int num_neighbors[MAX_POINTS];
    int num_points;
} DistanceMatrix;

/**
 * Calculates the cross product of two 3-dimensional vectors.
 *
 * @param a Input vector a (length 3)
 * @param b Input vector b (length 3)
 * @param result Output vector (length 3) to store the result
 */
void cross_product(const double a[3], const double b[3], double result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * Calculates the dot product of two 3-dimensional vectors.
 * 
 * @param a Input vector a (length 3)
 * @param b Input vector b (length 3)
 * @return Dot product of a and b
 */
void vec_axpy(double alpha, const double a[3], const double b[3], double c[3]) {
    // c = alpha * a + b
    cblas_dcopy(3, b, 1, c, 1);
    cblas_daxpy(3, alpha, a, 1, c, 1);
}

/**
 * Solve a system of three equations to find a point in 3D space.
 *
 * @param a 3D coordinates of the first point
 * @param b 3D coordinates of the second point
 * @param c 3D coordinates of the third point
 * @param da Distance from the unknown point to point a
 * @param db Distance from the unknown point to point b
 * @param dc Distance from the unknown point to point c
 * @param p Output: Projection of the solution on the plane formed by a, b, and c
 * @param w Output: Vector perpendicular to the plane
 */
void solveEQ3(const double a[3], const double b[3], const double c[3], const double da, const double db, const double dc, double p[3], double w[3]) {
    double u[3], v[3];
    double A11, A22, A12, A21, B[2], A[4];
    double uv, s;
    const double dtol = 1e-3;
    int ipiv[2];

    // Calculate u = b - a and v = c - a
    cblas_dcopy(3, b, 1, u, 1);
    cblas_daxpy(3, -1.0, a, 1, u, 1);
    cblas_dcopy(3, c, 1, v, 1);
    cblas_daxpy(3, -1.0, a, 1, v, 1);

    A11 = cblas_dnrm2(3, u, 1);
    A22 = cblas_dnrm2(3, v, 1);

    cblas_dscal(3, 1.0 / A11, u, 1);
    cblas_dscal(3, 1.0 / A22, v, 1);

    // Calculate w = u x v
    cross_product(u, v, w);
    cblas_dscal(3, 1.0 / cblas_dnrm2(3, w, 1), w, 1);

    uv = cblas_ddot(3, u, 1, v, 1);

    A12 = A11 * uv;
    A21 = A22 * uv;

    A[0] = A11; A[1] = A12; A[2] = A21; A[3] = A22;

    B[0] = (da * da - db * db + A11 * A11) / 2.0;
    B[1] = (da * da - dc * dc + A22 * A22) / 2.0;

    LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 1, A, 2, ipiv, B, 1);

    s = da * da - B[0] * B[0] - B[1] * B[1] - 2.0 * B[0] * B[1] * uv;

    if( s < 0 && fabs(s) > dtol ){
        printf("Error: the system is not solvable (s = %f)\n", s);
        exit(EXIT_FAILURE);
    }

    if (s < 0) {
        s = 0;
    }

    cblas_dcopy(3, a, 1, p, 1); // p = a
    cblas_daxpy(3, B[0], u, 1, p, 1); // p = a + y[0] * u
    cblas_daxpy(3, B[1], v, 1, p, 1); // p = p + y[1] * v

    cblas_dscal(3, sqrt(s), w, 1); // w = sqrt(s) * w
}

/**
 * Calculate the position and perpendicular vector for point i given its three parent points.
 *
 * @param i Index of the point to calculate
 * @param x Array of current point positions
 * @param D Distance matrix
 * @param p Output: Projection of the point on the plane formed by its parents
 * @param w Output: Vector perpendicular to the plane
 */
void calc_pw(int i, const double x[MAX_POINTS][3], const DistanceMatrix *D, double p[3], double w[3]) {
    
    const int ia = D->num_neighbors[i] - 3;
    const int ib = D->num_neighbors[i] - 2;
    const int ic = D->num_neighbors[i] - 1;

    const NeighborDistance *nda = &D->nd[i][ia];
    const NeighborDistance *ndb = &D->nd[i][ib];
    const NeighborDistance *ndc = &D->nd[i][ic];

    double da = nda->distance;
    double db = ndb->distance;
    double dc = ndc->distance;

    solveEQ3(x[nda->neighbor], x[ndb->neighbor], x[ndc->neighbor], da, db, dc, p, w);
}

/**
 * Check if the position of point i satisfies all distance constraints within the tolerance.
 *
 * @param D Distance matrix
 * @param i Index of the point to check
 * @param x Array of current point positions
 * @param dtol Tolerance for distance constraints
 * @return True if all distance constraints are satisfied, false otherwise
 */
bool is_feasible(const DistanceMatrix *D, int i, const double x[MAX_POINTS][3], double dtol) {
    double xi[3], xij[3], dij_eval, dij;
    int j, k;

    cblas_dcopy(3, x[i], 1, xi, 1);

    for (k = 0; k < D->num_neighbors[i]; k++) {
        j = D->nd[i][k].neighbor;
        dij = D->nd[i][k].distance;        
        
        // xij = x[j] - x[i]
        vec_axpy(-1.0, xi, x[j], xij);    

        // dij_eval = ||x[i] - x[j]||
        dij_eval = cblas_dnrm2(3, xij, 1);
        if (fabs(dij - dij_eval) > dtol) {
            return false;
        }
    }
    return true;
}

void assert_neighbor(const DistanceMatrix *D, int i, int j, int k) {
    if( D->nd[i][k].neighbor == j ){
        return;
    }
    printf("Error: %dth neighbor of %d is not %d\n", k, i, j);
    exit(EXIT_FAILURE);
}

/**
 * Initialize the first four points of the system.
 *
 * @param D Distance matrix
 * @param x Output: Array of initialized point positions
 * @param b Output: Array of solution choices for each point
 */
void init_xb(const DistanceMatrix *D, double x[MAX_POINTS][3], int b[MAX_POINTS]) {
    double d01, d02, d12, d03, d13, d23, cos_theta, sin_theta, p[3], w[3];

    assert_neighbor(D, 1, 0, 0);
    d01 = D->nd[1][0].distance;
    x[1][0] = d01;
    
    assert_neighbor(D, 2, 0, 0);
    d02 = D->nd[2][0].distance;

    assert_neighbor(D, 2, 1, 1);
    d12 = D->nd[2][1].distance;
    cos_theta = (d02 * d02 + d01 * d01 - d12 * d12) / (2 * d01 * d02);
    sin_theta = sqrt(1 - cos_theta * cos_theta);
    x[2][0] = d02 * cos_theta;
    x[2][1] = d02 * sin_theta;

    assert_neighbor(D, 3, 0, 0);
    d03 = D->nd[3][0].distance;

    assert_neighbor(D, 3, 1, 1);
    d13 = D->nd[3][1].distance;

    assert_neighbor(D, 3, 2, 2);
    d23 = D->nd[3][2].distance;

    solveEQ3(x[0], x[1], x[2], d03, d13, d23, p, w);
    
    // x[3] = p + w
    vec_axpy(1.0, p, w, x[3]);

    for (int i = 0; i < 4; i++) {
        b[i] = 1;
    }
}

int compare_nd(const void *a, const void *b) {
    NeighborDistance *na = (NeighborDistance *)a;
    NeighborDistance *nb = (NeighborDistance *)b;
    if (na->neighbor < nb->neighbor) {
        return -1;
    } else if (na->neighbor > nb->neighbor) {
        return 1;
    } else {
        return 0;
    }
}

void read_distance_matrix(const char *filename, DistanceMatrix *D, bool verbose) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        // Print error message and exit if file cannot be opened
        fprintf(stderr, "Error opening file: %s\n", filename);
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Initialize the distance matrix with a large value (assuming no self-loop)
    for (int i = 0; i < MAX_POINTS; i++) {
        D->num_neighbors[i] = 0;
        for (int j = 0; j < MAX_NEIGHBORS; j++) {
            D->nd[i][j].distance = -1;
            D->nd[i][j].neighbor = -1;
        }
    }

    int i = 0, j = 0;
    double dij = 0;
    D->num_points = 0;

    char line[256];
    char *token;

    // Skip the header line
    if (fgets(line, sizeof(line), file) == NULL) {
        perror("Error reading header");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    while (fgets(line, sizeof(line), file)) {
        token = strtok(line, ","); // Read i
        if (token) i = atoi(token);

        token = strtok(NULL, ","); // Read j
        if (token) j = atoi(token);

        // Skip next 4 columns
        for (int k = 0; k < 4; k++) {
            token = strtok(NULL, ",");
        }

        token = strtok(NULL, ","); // Read dij
        if (token) dij = atof(token);
        
        // Ensure i > j
        if( i < j ){
            // swap 
            int temp = i;
            i = j;
            j = temp;
        }

        D->nd[i][D->num_neighbors[i]].distance = dij;
        D->nd[i][D->num_neighbors[i]].neighbor = j;
        D->num_neighbors[i]++;

        if (i > D->num_points) D->num_points = i;
        if (j > D->num_points) D->num_points = j;
    }
    D->num_points++; // Adjust because indices are 0-based

    // sort all neighbors, using a library function
    for (int i = 0; i < D->num_points; i++) {
        qsort(D->nd[i], D->num_neighbors[i], sizeof(NeighborDistance), compare_nd);
    }

    if(verbose){
        printf("Number of points: %d\n", D->num_points);        
    }

    // check if the parents are correct
    for(int i = 3; i < D->num_points; ++i){
        int ia = D->num_neighbors[i] - 3;
        int ib = D->num_neighbors[i] - 2; 
        int ic = D->num_neighbors[i] - 1;

        if(D->nd[i][ia].neighbor != i - 3){
            printf("Error: %dth parent of %d is not %d\n", ia, i, i - 3);
            exit(EXIT_FAILURE);
        }

        if(D->nd[i][ib].neighbor != i - 2){
            printf("Error: %dth parent of %d is not %d\n", ib, i, i - 2);
            exit(EXIT_FAILURE);
        }

        if(D->nd[i][ic].neighbor != i - 1){
            printf("Error: %dth parent of %d is not %d\n", ic, i, i - 1);
            exit(EXIT_FAILURE);
        }
    }
    
    fclose(file);
}

bool bp(DistanceMatrix *D, int i, double x[MAX_POINTS][3], int b[MAX_POINTS], 
        bool single_solution, bool *finished, bool verbose) {
    if (i == D->num_points) {
        if(verbose){
            printf("Solution found!\n");
        }
        if (single_solution) {
            *finished = true;
        }
        return true;
    }

    double p[3], w[3];
    calc_pw(i, x, D, p, w);

    // Try positive direction
    vec_axpy(-1.0, w, p, x[i]);
    b[i] = 0;
    if (is_feasible(D, i, x, 1e-3) && i > 3) {
        if (bp(D, i + 1, x, b, single_solution, finished, verbose)) {
            if (*finished) {
                return true;
            }
        }
    }

    // Try negative direction
    vec_axpy(1.0, w, p, x[i]);
    b[i] = 1;
    if (is_feasible(D, i, x, 1e-3)) {
        if (bp(D, i + 1, x, b, single_solution, finished, verbose)) {
            if (*finished) {
                return true;
            }
        }
    }

    return false;
}

void set_output_filename(char *filename, char *output_filename) {
    char *prefix = "xbsol_leftmost";
    char *basename = strrchr(filename, '/');
    if (basename == NULL) {
        basename = filename;
    } else {
        basename++;
    }

    strcpy(output_filename, prefix);
    strcat(output_filename, "/");
    strcat(output_filename, basename);
}

void save_solution(const int num_points, const double x[MAX_POINTS][3], const int b[MAX_POINTS], char* output_filename, bool verbose) {
    if( verbose )
    {
        printf("Solution saved to %s\n", output_filename);
    }

    FILE *file = fopen(output_filename, "w");

    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", output_filename);
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write the header
    fprintf(file, "x,y,z,b\n");

    // Write the solution to the output file
    for (int i = 0; i < num_points; i++) {
        fprintf(file, "%f,%f,%f,%d\n", x[i][0], x[i][1], x[i][2], b[i]);
    }

    fclose(file);

}

void run_bp(DistanceMatrix *D, bool single_solution, char* filename, bool verbose) {
    double x[MAX_POINTS][3];
    int b[MAX_POINTS];
    bool finished = false;

    char output_filename[256];
    set_output_filename(filename, output_filename);

    // skip if the solution already exists
    if( access(output_filename, F_OK) != -1 ){
        if(verbose) printf("Solution already exists: %s\n", output_filename);        
        return;
    }

    init_xb(D, x, b);
    bp(D, 4, x, b, single_solution, &finished, verbose);

    if (finished) {
        save_solution(D->num_points, x, b, output_filename, verbose);
    } else {
        printf("No solution found\n");
    }
}

void read_option_bool(int argc, char *argv[], char *option, bool *value, bool default_value) {
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], option) == 0) {
            *value = true;
            return;
        }
    }
    *value = default_value;
}

int main(int argc, char *argv[]) {
    DistanceMatrix D;
    char filename[256];
    bool verbose = false;
    
    // Parse the input filename    
    if(argc > 1){
        strcpy(filename, argv[1]);
    } else {
        strcpy(filename, "dmdgp/1qfr_model1_chainA_segment8.csv");
    }
    
    read_option_bool(argc, argv, "-v", &verbose, false);

    // Read the distance matrix from the input file
    read_distance_matrix(filename, &D, verbose);

    // Print the distance matrix for verification
    if(verbose){
        for (int i = 0; i < D.num_points; i++) {
            for (int j = 0; j < D.num_neighbors[i]; j++) {
                printf("%d %d %f\n", i, D.nd[i][j].neighbor, D.nd[i][j].distance);
            }
            printf("\n");
        }
    }

    run_bp(&D, true, filename, verbose);

    return EXIT_SUCCESS;
}

