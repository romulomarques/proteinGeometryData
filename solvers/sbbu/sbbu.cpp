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
    double dtol = 1e-3;
    int imax = options.read_param("-imax", 1E9);
    bool verbose = options.read_param("-verbose", 0);
    bool fbs_active = options.read_param("-fbs", 0);
    bool dfs_all = options.read_param("-dfs_all", 0);
    std::string outdir = options.read_param("-outdir", "none");  

    dtol = options.read_param("-dtol", 1e-3);

    ddgp_t dgp(fname, dtol);
    sbbu_t sbbu(dgp, dtol, imax, dfs_all, fbs_active);
    sbbu.solve(tmax, fbs_active, verbose);
    
    // defining directory to save the results 
    sbbu.save_coords( fname, outdir, verbose );
    sbbu.save_edge_measurements( fname, outdir, verbose );
    
    return EXIT_SUCCESS;
}
