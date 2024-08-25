#pragma once
#include <stdexcept>

struct edge_t {
public:
   edge_t( int i, int j, double l, double u )
   {
      if ( l < u )
         throw std::invalid_argument( "Invalid edge (l > u)" );

      m_i = i;
      m_j = j;
      m_l = l;
      m_u = u;
   }

   bool operator<( const edge_t& v ) const
   {
      return ( m_i < v.m_i || ( m_i == v.m_i && m_j < v.m_j ) );
   }

   void show() const
   {
      printf( "(%d, %d, %g, %g, %s)\n", m_i, m_j, m_l, m_u, m_type );
   }

   int m_i; // first vertex of the edge
   int m_j; // second vertex of the edge
   double m_l; // distance lower bound
   double m_u; // distance upper bound
   char m_type[ 9 ] = {}; // string in the format "i_atom_type j-i j_atom_type" (e.g. "CA 1 H")
   int m_order = -1; // edge index in a "to-be-defined" edge ordering
};
