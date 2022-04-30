//#pragma once

#include <iostream>
#include <tuple>
#include <Eigen/Dense>

using namespace Eigen;
using std::cout;
using std::endl;

/**
 * @brief Returns nth derivative of B-splines from b-spline values
 * 
 * @param splines Array of length (order of splines) that contains all non zero B-splines for a point x
 * @param knots knot sequence of bsplines
 * @param index the index of the knot left of the point x
 * @param nthDeriv which derivative should be calculated, default = 1
 * @return ArrayXd
 */
ArrayXd ndxBsplines(ArrayXd splines, ArrayXd knotsInput, int index, uint nthDeriv=1)
{
    int kOrd = splines.size();


    ArrayXd knots(knotsInput.size()+2*kOrd-2);
    //we add the ghost points for calculation of the splines in every point
    knots << ArrayXd::Constant(kOrd-1, knotsInput(0)),knotsInput, ArrayXd::Constant(kOrd-1, knotsInput(last));

    ArrayXd deriv = splines;

    // this loop transforms the splines array into the correct derivative
    for (int n = nthDeriv; n > 0; n--)
    {
        // do the actual calculation

        int k = kOrd-n+1;
        int i = kOrd-1+index;

        deriv(kOrd-n) = -(kOrd-n)*deriv(kOrd-n-1) /(knots(i-kOrd+n+k)-knots(i-kOrd+n+1));

        for (int j = kOrd-n-1; j > 0; j--)
        {
            deriv(j) = (k-1)*deriv(j) / (knots(i-j+k-1)-knots(i-j))
                     - (k-1)*deriv(j-1) / (knots(i-j+k)-knots(i-j+1));
        }
        deriv(0) = (kOrd-n)*deriv(0) / (knots(i+k-1)-knots(i));

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
std::tuple<MatrixXd, int> bSplinesWithDeriv(double x, ArrayXd &knotsInput, int kOrd, int nDeriv)
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
            output.col(kOrd-k-1) = ndxBsplines(splines, knots, i, kOrd-k-1);
        }
    }

    output.col(0) = splines;
    return {output, i};

}

int main()
{
    int kOrd = 4;
    ArrayXd knots = ArrayXd::LinSpaced(11,0,1);

    double x = 0.52;
    int orderDeriv = 2;
    auto [mat, b] = bSplinesWithDeriv(x, knots, kOrd, orderDeriv);
    cout << mat << endl;
}