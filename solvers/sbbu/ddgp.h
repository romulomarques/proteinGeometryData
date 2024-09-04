#pragma once
#include "nmr.h"
#include <stdexcept>
#include <string>

class ddgp_t : public nmr_t
{
public:
   ddgp_t( std::string fname, double dtol )
       : nmr_t( fname, dtol, 1 )
   {
      // printf("%s::%d\n", __FILE__, __LINE__);

      m_a2 = (double*)calloc( m_nnodes, sizeof( double ) );
      m_b2 = (double*)calloc( m_nnodes, sizeof( double ) );
      m_c2 = (double*)calloc( m_nnodes, sizeof( double ) );
      double l, u;
      for ( int i = 0; i < m_nnodes; ++i )
      {
         // printf("%s::%d i = %d\n", __FILE__, __LINE__, i);
         if ( i > 2 )
         {
            get_bounds( i, i - 3, l, u );
            if ( fabs( l - u ) > 0 )
               throw std::invalid_argument( "The edge (" + std::to_string( i ) + ", " + std::to_string( i - 3 ) + ") is not exact." );
            m_a2[ i ] = l * l;
         }
         if ( i > 1 )
         {
            get_bounds( i, i - 2, l, u );
            if ( fabs( l - u ) > 0 )
               throw std::invalid_argument( "The edge (" + std::to_string( i ) + ", " + std::to_string( i - 2 ) + ") is not exact." );
            m_b2[ i ] = l * l;
         }
         if ( i > 0 )
         {
            get_bounds( i, i - 1, l, u );
            if ( fabs( l - u ) > 0 )
               throw std::invalid_argument( "The edge (" + std::to_string( i ) + ", " + std::to_string( i - 1 ) + ") is not exact." );
            m_c2[ i ] = l * l;
         }
      }
   }

   ~ddgp_t()
   {
      free( m_a2 );
      free( m_b2 );
      free( m_c2 );
   }

   /**
    * Mean Distance Error (MDE) function
    * @param x the vector of node coordinates
    * @return the mean distance error
    */
   double mde( const double* x )
   {
      int n = 0;      // number of terms
      double val = 0; // mde value
      for ( int i = 0; i < m_nnodes; ++i )
         for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         {
            const int j = m_j[ k ];
            if ( j > i )
               break;
            const double d = vec3_dist( &x[ 3 * i ], &x[ 3 * j ] );
            const double diff = fabs( m_l[ k ] - d );
            n++;
            val += ( m_l[ k ] > 0 ) ? diff / m_l[ k ] : diff;
         }      

      return val / n;
   }

   /**
    * Largest Distance Error (LDE) function
    * @param x the vector of node coordinates
    * @return the largest distance error
    */
   double lde( const double* x )
   {
      double emax = 0; // lde value
      for ( int i = 0; i < m_nnodes; ++i )
         for ( int k = m_i[ i ]; k < m_i[ i + 1 ]; ++k )
         {
            const int j = m_j[ k ];
            if ( j > i )
               break;
            const double dij = vec3_dist( &x[ 3 * i ], &x[ 3 * j ] );
            const double d = 0.5 * ( m_l[ k ] + m_u[ k ] );
            const double eij = fabs( dij - d );
            if ( eij > emax )
               emax = eij;
         }
      return emax;
   }

   double* m_a2;
   double* m_b2;
   double* m_c2;
};
