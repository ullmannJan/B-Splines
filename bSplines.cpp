#pragma once

#include <iostream>
#include <tuple>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using std::cout;
using std::endl;

template <typename T>
std::tuple<std::vector<T>, std::size_t> bsplines(T x,
                                                 const std::vector<T>& knots,
                                                 std::size_t order) {
  int n_knots = knots.size();

  // Check that input fulfills contrains, if in debug mode
#ifndef NDEBUG
  assert((knots[0] <= x) && (x <= knots[n_knots - 1]));
  for (int i = 1; i < n_knots; i++) {
    assert(knots[i - 1] <= knots[i]);
  }
#endif

  // function to get the knot, or automatically return ghost point if our\t of
  // range
  auto get_knot = [&knots, &n_knots](int i) {
    if (i < 0)
      return knots[0];
    else if (n_knots <= i)
      return knots[n_knots - 1];
    else
      return knots[i];
  };

  // Find the knot index where the value is
  int idx = 0;
  while (x > knots[idx + 1]) idx++;

  // initialize the vector with 0, except at the end
  auto iter = std::vector<T>(order + 1, 0);
  iter[order] = 1;

  // fill the vector from idx-cur_order to end (each iteration)
  // the last iteration will contain the non-zero b-spline
  // this algorithm is similar to DeBoor's algorithm
  // but does not calculate the sum, instead, it calculates
  // the single non-zero BSplines
  for (std::size_t cur_order = 1; cur_order <= order; cur_order++) {
    // this replaces the left 'ghost points'
    double w2 = (get_knot(idx + 1) - x) /
                (get_knot(idx + 1) - get_knot(idx - cur_order + 1));
    iter[order - cur_order] = w2 * iter[order - cur_order + 1];

    for (int i = idx - cur_order + 1; i < idx; i++) {
      const int iter_idx = i - idx + order;

      double w1 = (x - get_knot(i)) / (get_knot(i + cur_order) - get_knot(i));
      double w2 = (get_knot(i + cur_order + 1) - x) /
                  (get_knot(i + cur_order + 1) - get_knot(i + 1));

      // iteration
      iter[iter_idx] = w1 * iter[iter_idx] + w2 * iter[iter_idx + 1];
    }

    // this replaces the right 'ghost points'
    double w1 =
        (x - get_knot(idx)) / (get_knot(idx + cur_order) - get_knot(idx));
    // last iteration (iter_idx+1 will always be a ghost point or zero)
    iter[order] = w1 * iter[order];
  }

  // return the bsplines and the index
  return {iter, idx};
}

template <typename T>
std::tuple<std::vector<T>, std::size_t> ndx_bsplines(
    T x, const std::vector<T>& knots, std::size_t order,
    std::size_t nth_deriv = 1) {
  // derivative vanishes
  if (nth_deriv > order) {
    // Find the knot index where the value is
    int idx = 0;
    while (x > knots[idx + 1]) idx++;
    return {std::vector<T>(order + 1, 0), idx};
  }

  auto [iter, idx] = bsplines(x, knots, order - nth_deriv);

  int n_knots = knots.size();
  // function to get the knot, or automatically return ghost point if out of
  // range
  auto get_knot = [&knots, &n_knots](int i) {
    if (i < 0)
      return knots[0];
    else if (n_knots <= i)
      return knots[n_knots - 1];
    else
      return knots[i];
  };

  // resize the vector so we still have the spline of order
  // k-nth_deriv in the beginning, but now order+1 elements
  iter.resize(order + 1);

  for (int cur_order = order - nth_deriv + 1; cur_order <= int(order);
       cur_order++) {
    double w1 = 1. / (get_knot(idx + cur_order) - get_knot(idx));
    iter[cur_order] = cur_order * iter[cur_order - 1] * w1;

    // indices of previous iteration are shifted by 1
    for (int i = cur_order - 1; i > 0; i--) {
      const int real_i = idx - cur_order + i;
      double w1 = 1. / (get_knot(real_i + cur_order) - get_knot(real_i));
      double w2 =
          1. / (get_knot(real_i + cur_order + 1) - get_knot(real_i + 1));

      // indices of previous iteration are shifted by 1
      iter[i] = cur_order * (iter[i - 1] * w1 - iter[i] * w2);
    }

    double w2 = 1. / (get_knot(idx + 1) - get_knot(idx - cur_order + 1));
    iter[0] = -int(cur_order) * iter[0] * w2;
  }
  // return the bsplines derivatives and the index
  return {iter, idx};
}


/**
 * @brief Returns nth derivative of B-splines from b-spline values
 * 
 * @param splines Array of length (order of splines) that contains all non zero B-splines for a point x
 * @param knots knot sequence of bsplines
 * @param index the index of the knot left of the point x
 * @param nthDeriv which derivative should be calculated, default = 1
 * @return ArrayXd
 */
ArrayXd ndxBspline(ArrayXd splines, ArrayXd knots, int index, unsigned int nthDeriv=1)
{
    int knotsSize = knots.size();
    int kOrd = splines.size();
    auto getKnot = [&knots, &knotsSize](int i)
    {
        if (i < 0)
            return knots(0);
        else if (i >= knotsSize)
            return knots(knotsSize-1);
        else
            return knots(i);
    };

    ArrayXd deriv = splines;

    for (int n = nthDeriv; n > 0; n--)
    {
        int k = kOrd-n+1;
        int i = index+n;
        deriv(kOrd-n) = -(kOrd-n)*deriv(kOrd-n-1) /(getKnot(i+k)-getKnot(i+1));
        for (int j = n; j < kOrd-1; j++)
        {
            deriv(j) = (kOrd-n)*deriv(j) / (getKnot(i+j-n+1+k-1)-getKnot(i+j-n+1))
                     - (kOrd-n)*deriv(j+1) / (getKnot(i+j-n+1+k)-getKnot(i+j-n+1+1));
        }
        deriv(0) = (kOrd-n)*deriv(0) / (getKnot(i+k-1)-getKnot(i));
        //cout << deriv << endl;
    }

    return deriv;
}

/**
 * @brief Function that returns spline values at position x but only those that are not zero
 *        together with its 1st and 2nd derivatives and an index of which spline.
 *        Ghost points on both sides of the knots are automatically generated.
 * 
 * @param x The x where to retrieve the spline value;
 * @param knotsInput the knot points that have no ghost points yet
 * @param kOrd order of the splines that are calculated
 * @return MatrixXd that contains the values of the splines and derivs and the index i in the ghosted array
 *         remember that the splines are ordered as i, i-1, i-2, ... i-kOrd+1 in the arrays
 */
std::tuple<MatrixXd, int> bSplineAndDeriv(double x, ArrayXd &knotsInput, int kOrd, int nDeriv)
{
    MatrixXd output = MatrixXd::Zero(kOrd,1+nDeriv);
 
    //the last value hold the knot of the point, so that we have a reference to our splines
    ArrayXd splines = ArrayXd::Zero(kOrd);

    ArrayXd knots(knotsInput.size()+2*kOrd-2);
    //we add the ghost points for calculation of the splines in every point
    knots << ArrayXd::Constant(kOrd-1, knotsInput(0)),knotsInput, ArrayXd::Constant(kOrd-1, knotsInput(last));

    // we set the lowest layer BSplines manually
    splines(0) = 1;

    int i = kOrd-1; //we start at the first non ghost point

    //if x == knot we have to reduce the index manually by one
    if (x == knotsInput(last))
    {
        //special case is granted to the end ghost points as we want the first one here
        //we want the index where the last element starts, so no ghost points but the last real element
        i = knotsInput.size()-2+i;
    }
    else
    {
        //we search for the knot left of the x-value
        while (x >= knots(i))
        {
            i++;
        }
        i--;
    }
    //cout << "i-start: " << i << " knots(i): " << knots(i) << endl;

    //we iterate through to the k-2-th order in the triangular shape the problem
    //then we stop to calculate the derivative, after that we continue for 2 more iterations.
    for (int k = 1; k < kOrd; k++)
    {
        
        splines(k) = (knots(i+1)-x)/(knots(i+1)-knots(i-k+1))*splines(k-1);
        for (int j = 1; j < k; j++)
        {
            splines(k-j) = (x-knots(i-k+j)) / (knots(i+j)-knots(i-k+j)) * splines(k-j)
                        +(knots(i+j+1)-x) / (knots(i+j+1)-knots(i-k+j+1)) * splines(k-j-1);
        }
        splines(0) = (x-knots(i))/(knots(i+k)-knots(i))*splines(0);

        if (k > kOrd-nDeriv-2)
        {
            //cout << "spline\n" << splines << "\nk: " << k << endl;
            //output.col(kOrd-k-1) = ndxBspline(splines, knots, i, kOrd-k);
        }
    }

    output.col(0) = splines;
    return {output, i};

}

int main()
{
    std::vector<double> knots = {0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1};
    ArrayXd splines(4); 
    splines << 0.2,0.8,0,0;
    ArrayXd knots2 = ArrayXd::LinSpaced(11,0,1);

    double x = 0.52;
    auto [mat, b] = bSplineAndDeriv(x, knots2, 4, 2);
    cout << mat << endl;
    cout << ndxBspline(splines, knots2, 8,2) << "\n" << endl;

    auto [iter1, idx1] = bsplines(x, knots, 3);
    cout << "index" << idx1 << endl;
    for (auto x: iter1) cout << x << endl;
    
    auto [iter2, idx2] = ndx_bsplines(x, knots, 3,1);
    cout << "index" << idx2 << endl;
    for (auto x: iter2) cout << x << endl;

    auto [iter, idx] = ndx_bsplines(x, knots, 3,2);
    cout << "index" << idx << endl;
    for (auto x: iter) cout << x << endl;   
}