#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H

#include <cereal/archives/binary.hpp>

namespace Const {
    inline constexpr char32_t EOS = U'\x01';
    inline constexpr int STRINGS_NUM = 1.1e6;
    inline constexpr double eps = 1e-16;;
    inline constexpr char32_t MAX_CHAR = 0x10FFFF;
}


struct Idx_pos {int idx, pos;};
struct Pos_rk 
{
    int pos, rk;
    Pos_rk() {}
    Pos_rk(int x, int y): pos(x), rk(y) {}
    bool operator < (const Pos_rk& oth)const
    {
        return pos < oth.pos || (pos == oth.pos && rk < oth.rk);
    }

    template <class Archive>
    void serialize(Archive & archive) {
        archive(pos, rk);
    }
};
struct P
{
    int x; double y;
    P() {}
    P(int _x, double _y): x(_x), y(_y) {}
};
struct R_child 
{
    int R; char32_t c;
    R_child() {}
    R_child(int x, char32_t y): R(x), c(y) {}
    bool operator < (const R_child& oth)const
    {
        return R < oth.R || (R == oth.R && c < oth.c);
    }

    template <class Archive>
    void serialize(Archive & archive) {
        archive(R, c);
    }
};

inline int calc_len(char32_t* s)
{
    int m = 0;
    while (s[m] != U'\0') ++m;
    return m;
}

#endif // COMMON_STRUCTS_H