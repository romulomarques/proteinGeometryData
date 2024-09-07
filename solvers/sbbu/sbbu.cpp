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
    double dtol = 1e-7;
    int imax = options.read_param("-imax", 1E9);
    bool verbose = options.read_param("-verbose", 0);
    bool fbs_active = options.read_param("-fbs", 0);
    bool dfs_all = options.read_param("-dfs_all", 0);

    dtol = options.read_param("-dtol", 1e-3);

    ddgp_t dgp(fname, dtol);
    sbbu_t sbbu(dgp, dtol, imax, dfs_all, fbs_active);
    sbbu.solve(tmax, fbs_active, verbose);
    
    // defining directory to save the results 
    sbbu.save_coords( fname, fbs_active, verbose );
    sbbu.save_edge_measurements( fname, fbs_active, verbose );
    
    return EXIT_SUCCESS;
}
