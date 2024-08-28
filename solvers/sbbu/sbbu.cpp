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

    ddgp_t dgp(fname, dtol);
    sbbu_t sbbu(dgp, dtol, imax);   
    sbbu.solve(tmax);
    
    // defining directory to save the results
    std::string sol_dir = std::string("dmdgp_HA9H_sbbu");
    sbbu.save(fname, sol_dir);
    
    return EXIT_SUCCESS;
}
