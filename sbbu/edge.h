#pragma once
#include <stdexcept>

struct edge_t{
    public:
    edge_t(int i, int j, double l, double u){
        if( l < u )
        throw std::invalid_argument("Invalid edge (l > u)");

        // TODO Ensure that i < j
        m_i = i;
        m_j = j;
        m_l = l;
        m_u = u;
    }

    bool operator < (const edge_t& v) const
    {
        return (m_i < v.m_i || (m_i == v.m_i && m_j < v.m_j));
    }

    void show() const{
        printf("(%d, %d, %g, %g)\n", m_i, m_j, m_l, m_u);
    }

    int m_i;
    int m_j;
    double m_l;
    double m_u;
    char m_type[9] = {}; // string in the format "i_atom_name j-i j_atom_name"
};
