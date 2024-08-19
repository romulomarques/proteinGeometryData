#pragma once

#include "edge.h"
#include "vec3.h"
#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

class nmr_t
{
public:
   nmr_t() {}

   nmr_t( std::string fnmr, double dtol, bool is_csv )
   {
      if ( is_csv )
         read_csv( fnmr, dtol );
      else
         read_nmr( fnmr, dtol );
   }

   void read_nmr( std::string fnmr, double dtol )
   {
      m_fnmr = fnmr;
      m_dtol = dtol;
      printf( "Reading file %s\n", fnmr.c_str() );
      FILE* fid = fopen( fnmr.c_str(), "r" );
      if ( fid == NULL )
      {
         printf( "%s::%d The file %s could not be opened.\n", __FILE__, __LINE__, fnmr.c_str() );
         exit( EXIT_FAILURE );
      }

      int i, j, k = 0;
      double l, u;
      char iatom[ 3 ], jatom[ 3 ], iamino[ 4 ], jamino[ 4 ];
      m_nedges = m_nnodes = 0;

      // count nodes and edges
      while ( EOF != fscanf( fid, "%d %d %lf %lf %s %s %s %s\n", &i, &j, &l, &u, iatom, jatom, iamino, jamino ) )
      {
         m_nedges += 2; // add (i,j) and (j,i)
         m_nnodes = MAX( m_nnodes, i );
         m_nnodes = MAX( m_nnodes, j );
      }
      printf( "   NMR: nnodes = %d\n", m_nnodes );
      printf( "   NMR: nedges = %d\n", m_nedges );

      // read edges
      rewind( fid ); // back to file begin
      m_edges = (edge_t*)malloc( m_nedges * sizeof( edge_t ) );

      while ( EOF != fscanf( fid, "%d %d %lf %lf %s %s %s %s\n", &i, &j, &l, &u, iatom, jatom, iamino, jamino ) )
      {
         --i; // convert to zero-based
         --j; // convert to zero-based
         // add (i, j)
         m_edges[ k ].m_i = i;
         m_edges[ k ].m_j = j;
         m_edges[ k ].m_l = l;
         m_edges[ k++ ].m_u = u;

         // add (i, j)
         m_edges[ k ].m_i = j;
         m_edges[ k ].m_j = i;
         m_edges[ k ].m_l = l;
         m_edges[ k++ ].m_u = u;
      }
      fclose( fid );

      // set CSR idx
      std::sort( &m_edges[ 0 ], &m_edges[ m_nedges ] );

      m_i.assign( m_nnodes + 1, i = 0 );
      for ( k = 0; k < m_nedges; ++k )
         m_i[ m_edges[ k ].m_i + 1 ]++;
      for ( k = 0; k < m_nnodes; ++k )
         m_i[ k + 1 ] += m_i[ k ];

      // set CSR col and values
      m_j.resize( m_nedges );
      m_l.resize( m_nedges );
      m_u.resize( m_nedges );
      for ( k = 0; k < m_nedges; ++k )
      {
         m_j[ k ] = m_edges[ k ].m_j;
         m_l[ k ] = m_edges[ k ].m_l;
         m_u[ k ] = m_edges[ k ].m_u;
      }
   }

   void read_csv( std::string fcsv, double dtol )
   {
      m_fnmr = fcsv;
      m_dtol = dtol;
      printf( "Reading file %s\n", fcsv.c_str() );
      FILE* fid = fopen( fcsv.c_str(), "r" );
      if ( fid == NULL )
      {
         printf( "%s::%d The file %s could not be opened.\n", __FILE__, __LINE__, fcsv.c_str() );
         exit( EXIT_FAILURE );
      }

      int i, j, k = 0;
      char iatom[ 3 ], jatom[ 3 ];
      int iresnumber, jresnumber;
      double l, u;
      m_nedges = m_nnodes = 0;

      // the first line of the csv file contains the names of the columns and must be ignored
      int nreads = fscanf( fid, "%*[^\n]\n" );
      if ( nreads == EOF )
      {
         printf( "%s::%d The file %s is empty.\n", __FILE__, __LINE__, fcsv.c_str() );
         exit( EXIT_FAILURE );
      }

      // count nodes and edges.
      while ( EOF != fscanf( fid, "%d,%d,%[^,],%[^,],%d,%d,%lf,%*[^\n]\n", &i, &j, iatom, jatom, &iresnumber, &jresnumber, &l ) )
      {
         m_nedges += 2; // add (i,j) and (j,i)
         m_nnodes = MAX( m_nnodes, i );
         m_nnodes = MAX( m_nnodes, j );
      }
      printf( "   NMR: nnodes = %d\n", m_nnodes );
      printf( "   NMR: nedges = %d\n", m_nedges );

      // read edges
      rewind( fid ); // back to file begin

      // the first line of the csv file contains the names of the columns and must be ignored
      nreads = fscanf( fid, "%*[^\n]\n" );
      if ( nreads == EOF )
      {
         printf( "%s::%d The file %s is empty.\n", __FILE__, __LINE__, fcsv.c_str() );
         exit( EXIT_FAILURE );
      }

      m_edges = (edge_t*)malloc( m_nedges * sizeof( edge_t ) );

      char edge_type[ 9 ];
      int diff_ji;
      while ( EOF != fscanf( fid, "%d,%d,%[^,],%[^,],%d,%d,%lf,%*[^\n]\n", &i, &j, iatom, jatom, &iresnumber, &jresnumber, &l ) )
      {
         u = l; // the files contain just one exact distance per row.

         // keeping edges increasingly ordered
         if ( i > j ) SWAP( i, j );

         // cleaning the array 'edge_type'
         strcpy( edge_type, "" );

         // setting the edge type
         diff_ji = j - i;
         sprintf( edge_type, "%s %d %s", iatom, diff_ji, jatom );
         printf( "%s\n", edge_type );

         // add (i, j)
         m_edges[ k ].m_i = i;
         m_edges[ k ].m_j = j;
         m_edges[ k ].m_l = l;
         m_edges[ k ].m_u = u;
         strcpy( m_edges[ k++ ].m_type, edge_type );

         // add (i, j)
         m_edges[ k ].m_i = j;
         m_edges[ k ].m_j = i;
         m_edges[ k ].m_l = l;
         m_edges[ k ].m_u = u;
         strcpy( m_edges[ k++ ].m_type, edge_type );
      }
      fclose( fid );

      // set CSR idx
      std::sort( &m_edges[ 0 ], &m_edges[ m_nedges ] );

      m_i.assign( m_nnodes + 1, i = 0 );
      for ( k = 0; k < m_nedges; ++k )
         m_i[ m_edges[ k ].m_i + 1 ]++;
      for ( k = 0; k < m_nnodes; ++k )
         m_i[ k + 1 ] += m_i[ k ];

      // set CSR col and values
      m_j.resize( m_nedges );
      m_l.resize( m_nedges );
      m_u.resize( m_nedges );
      for ( k = 0; k < m_nedges; ++k )
      {
         m_j[ k ] = m_edges[ k ].m_j;
         m_l[ k ] = m_edges[ k ].m_l;
         m_u[ k ] = m_edges[ k ].m_u;
      }
   }

   bool feasible( const double* x )
   {
      for ( int i = 0; i < m_nnodes; ++i )
         for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         {
            const int j = m_j[ k ];
            if ( j > i )
               break;
            const double d = vec3_dist( &x[ 3 * i ], &x[ 3 * j ] );
            if ( ( d < ( m_l[ k ] - m_dtol ) ) || ( d > ( m_u[ k ] + m_dtol ) ) )
               return false;
         }
      return true;
   }

   void assert_feasibility( const double* x )
   {
      // TODO DRY this code by using feasible() method
      for ( int i = 0; i < m_nnodes; ++i )
         for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         {
            const int j = m_j[ k ];
            if ( j > i )
               break;
            const double d = vec3_dist( &x[ 3 * i ], &x[ 3 * j ] );
            const double err_l = m_l[ k ] - d;
            const double err_u = d - m_u[ k ];
            if ( ( err_l > m_dtol ) || ( err_u > m_dtol ) )
            {
               char msg[ 256 ];
               printf( "m_dtol = %E\n", m_dtol );
               snprintf( msg, sizeof( msg ), "NMR: The input is not a solution (err_l=%g, err_u=%g)", err_l, err_u );
               throw std::runtime_error( msg );
            }
         }
   }

   bool feasible( const double* x, int i )
   {
      if ( i > ( m_nnodes - 1 ) )
      {
         printf( "%s::%d invalid index.\n", __FILE__, __LINE__ );
         exit( EXIT_FAILURE );
      }

      for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
      {
         const int j = m_j[ k ];
         if ( j > i )
            break;
         const double d = vec3_dist( &x[ 3 * i ], &x[ 3 * j ] );
         if ( ( d < ( m_l[ k ] - m_dtol ) ) || ( d > ( m_u[ k ] + m_dtol ) ) )
            return false;
      }
      return true;
   }

   double get_l( int i, int j )
   {
      if ( i < j )
      {
         const int t = i;
         i = j;
         j = t;
      }
      for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         if ( j == m_j[ k ] )
            return m_l[ k ];
      // invalid edge
      throw std::invalid_argument( "There is no edge (" + std::to_string( i ) + ", " + std::to_string( j ) + ")." );
   }

   double get_u( int i, int j )
   {
      if ( i < j )
      {
         const int t = i;
         i = j;
         j = t;
      }
      for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         if ( j == m_j[ k ] )
            return m_u[ k ];
      // invalid edge
      throw std::invalid_argument( "There is no edge (" + std::to_string( i ) + ", " + std::to_string( j ) + ")." );
   }

   void get_bounds( int i, int j, double& l, double& u )
   {
      if ( i < j )
      {
         const int t = i;
         i = j;
         j = t;
      }
      for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         if ( j == m_j[ k ] )
         {
            l = m_l[ k ];
            u = m_u[ k ];
            return;
         }
      throw std::invalid_argument( "There is no edge (" + std::to_string( i ) + ", " + std::to_string( j ) + ")." );
   }

   virtual ~nmr_t()
   {
      free( m_edges );
      m_edges = NULL;
   }

   double m_dtol;
   int m_nnodes;
   int m_nedges;
   std::vector<int> m_i;
   std::vector<int> m_j;
   std::vector<double> m_l;
   std::vector<double> m_u;
   edge_t* m_edges;
   std::string m_fnmr;
};
