#include "common_structs.h"
#include <vector>
#include <algorithm>
#include <assert.h>
#include <cereal/archives/binary.hpp>


struct Spline
{
    int pos; 
    double k, b;
    bool operator < (const Spline& oth)const
    {
        return pos < oth.pos || (pos == oth.pos && k < oth.k);
    }

    template <class Archive>
    void serialize(Archive & archive) {
        archive(pos, k, b);
    }
};

class GreedySpline
{
private:
    int SUM_DATA_LEN;
    int error_bound;
    std::vector<Pos_rk>& pr;
    int cur_num_keys = 0, curr_num_distinct_keys_ = 0;
    int prev_key_, prev_position_;
    std::vector<P> spline_points_;
    P prev_point_, upper_limit_, lower_limit_;
public:
    GreedySpline(int SUM_DATA_LEN, int e, std::vector<Pos_rk>& pr): SUM_DATA_LEN(SUM_DATA_LEN), error_bound(e), pr(pr)
    {
        cur_num_keys = curr_num_distinct_keys_ = 0;
        spline_points_.clear();
    }

    int ComputeOrientation(double dx1, double dy1, double dx2, double dy2)
    {
        double expr = dy1 * dx2 - dy2 * dx1;
        if(expr > Const::eps) return 0;
        else if(expr < -Const::eps) return 2;
        else return 1;
    }

    void addKeytoSpline(int x, int y)
    {
        if(cur_num_keys == 0)
        {
            spline_points_.push_back(P(x, y));
            curr_num_distinct_keys_ += 1;
            prev_point_ = P(x, y);
            return;
        }
        if(x == prev_key_) return;
        
        curr_num_distinct_keys_ += 1;
        
        if(curr_num_distinct_keys_ == 2)
        {
            upper_limit_ = P(x, y + error_bound);
            lower_limit_ = P(x, std::max(0, y - error_bound));
            prev_point_ = P(x, y);
            return;
        }

        P las = spline_points_.back();
        double upper_y = y + error_bound;
        double lower_y = std::max(0, y - error_bound);

        assert(upper_limit_.x >= las.x);
        assert(lower_limit_.x >= las.x);
        assert(x >= las.x);

        int upper_limit_x_diff = upper_limit_.x - las.x;
        int lower_limit_x_diff = lower_limit_.x - las.x;
        int x_diff = x - las.x;

        assert(upper_limit_.y >= las.y);
        assert(y >= las.y);
        double upper_limit_y_diff = upper_limit_.y - las.y;
        double lower_limit_y_diff = lower_limit_.y - las.y;
        double y_diff = y - las.y;

        assert(prev_point_.x != las.x);

        if(ComputeOrientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, y_diff) != 0 ||
            ComputeOrientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, y_diff) != 2)
        {
                spline_points_.push_back(prev_point_);
                upper_limit_ = P(x, upper_y);
                lower_limit_ = P(x, lower_y);
        }
        else
        {
            assert(upper_y >= las.y);
            double upper_y_diff = upper_y - las.y;
            if(ComputeOrientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, upper_y_diff) == 0)
                upper_limit_ = P(x, upper_y);
            double lower_y_diff = lower_y - las.y;
            if(ComputeOrientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, lower_y_diff) == 2)
                lower_limit_ = P(x, lower_y);
        }
        prev_point_ = P(x, y);
    }

    Spline calc_kb(int x1, double y1, int x2, double y2)
    {
        double k = (y2 - y1) / (x2 - x1);
        double b = y2 - k * x2;
        assert(k >= 0);
        return (Spline){x2, k, b};
    }


    std::vector<Spline> build_greedyspline()
    {
        if(pr.size() == 1)
        {
            std::vector<Spline> ret{(Spline){pr[0].pos, 0, 1.0 * pr[0].rk}};
            return ret;
        }
        for(int i = 0; i < (int)pr.size(); ++i)
        {
            addKeytoSpline(pr[i].pos, pr[i].rk);
            cur_num_keys += 1;
            prev_key_ = pr[i].pos;
            prev_position_ = pr[i].rk;
        }
        spline_points_.push_back(P(pr.back().pos, pr.back().rk));
        std::vector<Spline> ret; ret.clear();
        for(int i = 1; i < (int)spline_points_.size(); ++i)
        {
            Spline p = calc_kb(spline_points_[i - 1].x, spline_points_[i - 1].y, spline_points_[i].x, spline_points_[i].y);
            assert(p.k >= 0);
            assert(p.pos <= SUM_DATA_LEN);
            ret.push_back(p);
        }
        return ret;
    }
};



