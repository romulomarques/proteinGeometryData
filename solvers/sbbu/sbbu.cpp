#include <cstdlib>
#include <cstdio>
#include <string>
#include <cmath>
#include "ddgp.h"
#include "sbbu.h"
#include "utils.h"

int main(int argc, char *argv[])
{    
    options_t options(argc, argv);

    std::string fname = options.read_param("-nmr", "none");
    double tmax = options.read_param("-tmax", 10);
    double dtol = options.read_param("-dtol", 1e-3);
    int imax = options.read_param("-imax", 1E9);
    std::string order = options.read_param("-order", "default");

    ddgp_t dgp(fname, dtol);
    sbbu_t sbbu(dgp, dtol, imax);
    sbbu.solve(tmax, order);
    sbbu.save(fname);
    return EXIT_SUCCESS;
}
