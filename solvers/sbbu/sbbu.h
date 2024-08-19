#pragma once

#include "ddgp.h"
#include "qs.h"
#include <omp.h>
#include <string.h>
#include <string>

struct cluster_t {
public:
   double* m_d;  // decision vector
   double* m_a2; // vector of squared distances to x[i-1]
   double* m_b2; // vector of squared distances to x[i-2]
   double* m_c2; // vector of squared distances to x[i-3]
   ddgp_t* m_dgp;
   int m_i;
   int m_j;
   int m_nnodes;
   double m_dtol;
   double m_y[ 3 ];
   double* m_x;
   int m_m;     // number of planes
   int m_k;     // plane capacity
   double* m_a; // plane point
   double* m_n; // plane normal
   double* m_w; // plane constant

   cluster_t( ddgp_t& dgp, int v )
       : m_dtol( 1e-7 )
   {
      m_dgp = &dgp;
      m_nnodes = dgp.m_nnodes;
      m_i = v; // cluster begining
      m_j = v; // cluster ending
      m_d = &dgp.m_l[ 0 ];
      m_x = m_a = m_n = m_w = NULL;
      m_k = m_m = 0;
   }

   ~cluster_t()
   {
      free( m_a );
      free( m_n );
      free( m_w );
   }

   int size()
   {
      return m_j - m_i + 1;
   }

   // Sets the reflection planes and
   void create_planes( int j, int* d, int n )
   {
      m_j = j;
      // set y == m_x[j], point that must be reflected (last cluster point)
      vec3_copy( m_y, &m_x[ 3 * m_j ] );
      // create decision reflection planes
      m_m = 0; // number of planes
      for ( int k = 0; k < n; ++k )
      {
         const double* a = &m_x[ 3 * ( d[ k ] - 1 ) ];
         const double* b = &m_x[ 3 * ( d[ k ] - 2 ) ];
         const double* c = &m_x[ 3 * ( d[ k ] - 3 ) ];
         add_plane( a, b, c );
         ++m_m;
      }
   }

   // Adds the plane that passes through a, b, c
   void add_plane( const double* a, const double* b, const double* c )
   {
      double U[ 3 ], V[ 3 ];

      vec3_axpy( U, -1.0, a, b );
      vec3_axpy( V, -1.0, a, c );

      double* A = &m_a[ 3 * m_m ];
      double* N = &m_n[ 3 * m_m ];

      vec3_copy( A, a);
      vec3_cross( N, U, V );
      vec3_unit( N );
      m_w[ m_m ] = vec3_dot( A, N );      
   }

   // n = len(f)
   inline void reflect( bool* f, int n, double *xj)
   {      
      // reset x[j,:]
      vec3_copy( xj, m_y );
      
      // apply from last to first
      for ( int k = n - 1; k >= 0; --k )
         if ( f[ k ] )
            mirror( k, xj );
   }

   // n = len(f) = len(d)
   inline void reflect_all( int* d, bool* f, int n , int j )
   {
      // reset x[j,:]
      vec3_copy( &m_x[3 * j], m_y );
      // apply from last to first
      for ( int k = n - 1; k >= 0; --k )
         if ( f[ k ] )
            for ( int i = d[ k ]; i <= m_j; ++i )
               mirror( k, &m_x[ 3 * i ] );
   }

   inline void mirror( int k, double* x )
   {
      const double* N = &m_n[ 3 * k ];
      const double q = 2 * ( vec3_dot( x, N ) - m_w[ k ] );
      x[ 0 ] -= q * N[ 0 ];
      x[ 1 ] -= q * N[ 1 ];
      x[ 2 ] -= q * N[ 2 ];
   }
};

class sbbu_t
{
public:
   int m_nnodes;
   int m_nedges;
   int m_n;           // n :: len(d)
   int* m_d;          // decision vector
   int* m_root;       // cluster roots
   bool* m_f;         // flip vector
   bool* m_fopt;      // optimal flip vector
   double* m_x;       // solution vector (euclidean)
   double* m_plane_a; // plane ref points
   double* m_plane_n; // plane normals
   double* m_plane_w; // plane constants
   int m_imax;        // max number of inner iterations
   ddgp_t& m_dgp;
   edge_t* m_edges;
   cluster_t* m_c;
   double m_dtol;
   int m_j; // last solved node

   sbbu_t( ddgp_t& dgp, double dtol, int imax )
       : m_dgp( dgp )
   {
      m_nnodes = dgp.m_nnodes;
      m_dtol = dtol;
      m_imax = imax;
      m_d = (int*)malloc( m_nnodes * sizeof( int ) );
      m_f = (bool*)malloc( m_nnodes * sizeof( bool ) );
      m_fopt = (bool*)malloc( m_nnodes * sizeof( bool ) );
      m_plane_n = (double*)malloc( 3 * dgp.m_nnodes * sizeof( double ) );
      m_plane_a = (double*)malloc( 3 * dgp.m_nnodes * sizeof( double ) );
      m_plane_w = (double*)malloc( dgp.m_nnodes * sizeof( double ) );

      // init (prune) edges
      init_edges();

      // init m_x;
      m_x = (double*)malloc( 3 * m_nnodes * sizeof( double ) );
      init_x();

      // init m_c
      m_c = (cluster_t*)malloc( m_nnodes * sizeof( cluster_t ) );
      for ( int k = 0; k < m_nnodes; ++k )
         init_c( k, dgp );

      // init m_root
      m_root = (int*)malloc( m_nnodes * sizeof( int ) );
      for ( int k = 0; k < m_nnodes; ++k )
         m_root[ k ] = -1;
   }

   ~sbbu_t()
   {
      free( m_edges );
      free( m_root );
      free( m_d );
      free( m_f );
      free( m_c );
      free( m_x );
      free( m_plane_a );
      free( m_plane_n );
      free( m_plane_w );
   }

   void init_x()
   {
      double d01 = m_dgp.get_l( 0, 1 );
      double d02 = m_dgp.get_l( 0, 2 );
      double d12 = m_dgp.get_l( 1, 2 );
      double x20 = ( d02 * d02 - d12 * d12 + d01 * d01 ) / ( 2 * d01 );
      double x21 = sqrt( d02 * d02 - x20 * x20 );
      vec3_set( &m_x[ 0 ], 0., 0., 0. );
      vec3_set( &m_x[ 3 ], d01, 0., 0. );
      vec3_set( &m_x[ 6 ], x20, x21, 0. );
      m_j = 2;
   }

   // init m_x[m_j+1:j] using BP distances
   inline void init_x( const int j )
   {
      // forward to set x
      for ( int i = m_j + 1; i <= j; ++i )
         fwd_x( i );
      m_j = j;
   }

   // Set m_x[i] given points A, B, C and their distances a2, b2, c2.
   // m_x[i] = P + alpha * N, where N is the normal to the plane A,B,C and P is the proj of m_x[i] on this plane.
   void set_x( int i, const double* A, double a2, const double* B, double b2, const double* C, double c2 )
   {
      const double err_a = fabs( vec3_dist2( A, &m_x[ 3 * i ] ) - a2 );
      const double err_b = fabs( vec3_dist2( B, &m_x[ 3 * i ] ) - b2 );
      const double err_c = fabs( vec3_dist2( C, &m_x[ 3 * i ] ) - c2 );

      // m_x[i] does not need to be moved
      if ( err_a < m_dtol && err_b < m_dtol && err_c < m_dtol )
         return;

      double N[ 3 ], U[ 3 ], V[ 3 ], P[ 3 ];
      vec3_axpy( U, -1.0, A, B );
      vec3_axpy( V, -1.0, A, C );
      vec3_cross( N, U, V );
      vec3_unit( N );

      qs_p( P, N, A, a2, B, b2, C, c2, m_dtol );

      double beta = vec3_dist2( P, A );
      if ( a2 + m_dtol < beta )
         throw std::runtime_error( "The step size could not be calculated." );
      double alpha = sqrt( fabs( a2 - beta ) );

      // x[i] = P + alpha * N
      vec3_axpy( &m_x[ 3 * i ], alpha, N, P );

#ifdef DEBUG
      if ( fabs( vec3_dist2( &m_x[ 3 * i ], A ) - a2 ) > m_dtol || fabs( vec3_dist2( &m_x[ 3 * i ], B ) - b2 ) > m_dtol || fabs( vec3_dist2( &m_x[ 3 * i ], C ) - c2 ) > m_dtol )
         throw std::runtime_error( "A solution could not be found." );
#endif
   }

   inline void bck_x( int i )
   {
      const double* A = &m_x[ 3 * ( i + 1 ) ];
      const double* B = &m_x[ 3 * ( i + 2 ) ];
      const double* C = &m_x[ 3 * ( i + 3 ) ];
      set_x( i, A, m_dgp.m_a2[ i + 1 ], B, m_dgp.m_b2[ i + 2 ], C, m_dgp.m_c2[ i + 3 ] );
   }

   inline void fwd_x( int i )
   {
      const double* A = &m_x[ 3 * ( i - 1 ) ];
      const double* B = &m_x[ 3 * ( i - 2 ) ];
      const double* C = &m_x[ 3 * ( i - 3 ) ];
      set_x( i, A, m_dgp.m_a2[ i ], B, m_dgp.m_b2[ i ], C, m_dgp.m_c2[ i ] );
   }

   void init_c( int k, ddgp_t& dgp )
   {
      m_c[ k ].m_i = k;
      m_c[ k ].m_j = k;
      m_c[ k ].m_k = 0;
      m_c[ k ].m_x = m_x;
      m_c[ k ].m_dgp = &dgp;
      m_c[ k ].m_a2 = dgp.m_a2;
      m_c[ k ].m_b2 = dgp.m_b2;
      m_c[ k ].m_c2 = dgp.m_c2;
      m_c[ k ].m_nnodes = dgp.m_nnodes;
      m_c[ k ].m_dtol = 1e-7;
      m_c[ k ].m_a = m_plane_a;
      m_c[ k ].m_n = m_plane_n;
      m_c[ k ].m_w = m_plane_w;
   }

   void init_edges()
   {
      // counting prunning edges
      m_nedges = 0;
      for ( int k = 0; k < m_dgp.m_nedges; ++k )
      {
         edge_t& edge = m_dgp.m_edges[ k ];
         if ( edge.m_j - edge.m_i > 3 )
            m_nedges++;
      }
      m_edges = (edge_t*)malloc( m_nedges * sizeof( edge_t ) );
      m_nedges = 0;
      for ( int k = 0; k < m_dgp.m_nedges; ++k )
      {
         edge_t& edge = m_dgp.m_edges[ k ];
         if ( edge.m_j - edge.m_i > 3 )
            m_edges[ m_nedges++ ] = edge;
      }      
   }

   // Returns the (index) root associated to the vertex i cluster.
   int find_root( int i )
   {
      // v :: list
      // i :: start index
      int s = i, r = i;
      while ( m_root[ r ] >= 0 )
      {
         m_root[ s ] = m_root[ r ];
         s = r;
         r = m_root[ r ];
      }
      return r;
   }

   // Merges root[r] += root[i]
   void merge_cluster( int i, int r )
   {
      if ( r == i )
         return;
      // increment the number of nodes in the cluster
      m_root[ r ] += m_root[ i ];
      // set the root of the i-th node to r
      m_root[ i ] = r;
   }

   void save( std::string fname )
   {
      char fsol[ FILENAME_MAX ];
      strcpy( fsol, fname.c_str() );
      char* p = strstr( fsol, ".csv" ); // returns a pointer to the first occurrence of ".nmr"
      sprintf( p, "_sbbu.sol" );        // replace suffix

      printf( "SBBU: saving solution on %s\n", fsol );
      FILE* fid = fopen( fsol, "w" );
      if ( fid == NULL )
         throw std::runtime_error( "The solution file could not be created." );
      for ( auto k = 0; k < m_nnodes; ++k )
         fprintf( fid, "%.18g %.18g %.18g\n", m_x[ 3 * k ], m_x[ 3 * k + 1 ], m_x[ 3 * k + 2 ] );
      fclose( fid );
   }

   double dfs_traverse( const edge_t& edge, cluster_t& cr )
   {
      const int kmax = m_n - 1;
      double* xi = &m_x[ 3 * edge.m_i ];
      double* xj = &m_x[ 3 * edge.m_j ];

      double eij = 0.0;
      int niters = 0;

      for ( int k = kmax, count = 0; count < m_imax; ++count )
      {
         eij = fabs( vec3_dist( xi, xj ) - edge.m_l );

         // solution found
         if ( eij < m_dtol )
         {
            // update the best solution
            for ( int i = 0; i <= kmax; ++i )
               m_fopt[ i ] = m_f[ i ];
            break;
         }

         if ( k == kmax ) // backtrack
         {
            for ( ; k >= 0 && m_f[ k ]; --k )
               m_f[ k ] = false;
            if ( k < 0 )
               break;
            m_f[ k ] = true;
         }
         else
            ++k;
         cr.reflect( m_f, m_n, xj ); // updates x
         ++niters;
      }

      return eij;
   }

   double fbs_traverse(){
      double eij = 0.0;
      return eij;
   }

   void solve_edge( const edge_t& edge )
   {
      // init x[k,:], for k in edge.i, edge.i+1, ..., m_j[m_j[edge.i]]
      int r = find_root( edge.m_i + 3 );
      int j = find_root( edge.m_j );
      if ( r == j ) // already solved
      {
         // calculate the distance between x[i] and x[j]
         double* xi = &m_x[ 3 * edge.m_i ];
         double* xj = &m_x[ 3 * edge.m_j ];
         double eij = vec3_dist( xi, xj ) - edge.m_l;
         if ( fabs( eij ) > m_dtol )
         {
            char msg[ 256 ];
            snprintf( msg, sizeof( msg ), "The edge (%d, %d, %f) could not be solved (emin=%g).\n",
                edge.m_i + 1, edge.m_j + 1, edge.m_l, eij );
            throw std::runtime_error( msg );
         }
         return;
      }

      init_x( edge.m_j );
      cluster_t& cr = m_c[ r ];      

      // create d :: vector of decisions
      m_n = 0; // number of decisions to be taken
      cluster_t* ck = &cr;
      for ( int k = r; ck->m_i <= edge.m_j; )
      {
         merge_cluster( k, r );
         // add to decision vector
         m_d[ m_n ] = k;
         // reset the flip vector
         m_f[ m_n ] = false;
         ++m_n;
         
         // last feasible cluster
         if ( ck->m_j + 1 == m_nnodes )
            break;
         // next cluster
         k = find_root( ck->m_j + 1 );
         ck = &m_c[ k ];
      }
      cr.create_planes( edge.m_j, m_d, m_n );

      // searching      
      double eij = dfs_traverse( edge, cr );

      // reflect all nodes from edge.i to edge.j
      cr.reflect_all( m_d, m_fopt, m_n, m_j );
      
      if ( eij > m_dtol ) // edges of range 4 allways have two solutions
      {
         char msg[ 256 ];
         snprintf( msg, sizeof( msg ), "The edge (%d, %d, %f) could not be solved (emin=%g, len(m_d)=%d).\n",
             edge.m_i + 1, edge.m_j + 1, edge.m_l, eij, m_n );
         throw std::runtime_error( msg );
      }
   }

   void sort_edges_default()
   {
      // edge a, edge b
      // a < b, if a.j < b.j or (a.j == b.j and a.i > b.i)
      auto cmp_edges = []( const void* x, const void* y ) {
         // j is different
         auto dj = ( (edge_t*)x )->m_j - ( (edge_t*)y )->m_j;
         if ( dj ) return dj;

         // j is equal return the edge with small length
         auto dx = ( (edge_t*)x )->m_j - ( (edge_t*)x )->m_i;
         auto dy = ( (edge_t*)y )->m_j - ( (edge_t*)y )->m_i;
         return dx - dy;
      };

      qsort( m_edges, m_nedges, sizeof( edge_t ), cmp_edges );
   }

   void sort_edges( const std::string& order )
   {
      if ( order == "default" )
         sort_edges_default();
      else
         throw std::invalid_argument( "Invalid order (" + order + ")" );
   }

   void solve( double tmax, const std::string& order )
   {
      printf( "SBBU: tmax = %g\n", tmax );
      printf( "SBBU: dtol = %g\n", m_dtol );
      printf( "SBBU: imax = %g\n", (double)m_imax );
      printf( "SBBU: prune_edges = %d\n", m_nedges );

      sort_edges( order );

      double tic = omp_get_wtime();
      for ( int k = 0; k < m_nedges; ++k )
      {
         solve_edge( m_edges[ k ] );
         if ( omp_get_wtime() - tic > tmax )
            throw std::runtime_error( "SBBU: time exceeded (tmax = " + std::to_string( tmax ) + ")." );
      }      
      double toc = omp_get_wtime() - tic;
      printf( "SBBU: solution found after %g secs\n", toc );

      init_x( m_nnodes - 1 );
      m_dgp.assert_feasibility( m_x );

      printf( "SBBU: MDE = %g, LDE = %g\n", m_dgp.mde( m_x ), m_dgp.lde( m_x ) );
   }
};
