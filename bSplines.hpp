/**
 * @file bSplines.hpp
 * @author Ludwig Neste & Jan Ullmann
 * @brief Implemenation of an efficient algorithm to calculate the values of
 * BSplines and their derivatives in modern C++ using the Eigen package.
 * (C++ 17 required, compile with `-std=c++17`)
 *
 * The used algorithms are similar to and inspired by DeBoors algorithm
 * (https://en.wikipedia.org/wiki/De_Boor%27s_algorithm), but distinct.
 *
 * The useful function is called bSplinesWithDeriv() and is the one that should be used.
 * 
 * Both functions provide some code example, to give you an idea how to
 * use them.
 *
 *
 */

#pragma once

#include <iostream>
#include <tuple>
#include <Eigen/Dense>

using namespace Eigen;
using std::cout;
using std::endl;

/**
 * @brief calculates non-zero n-th derivative of B-Spline B_k,i(x) from the non-zero B-splines B_k-n,i(x). 
 *        The splines are ordered in descending order e.g. [B_i, B_i-1, B_i-2, ... , B_i-kOrd+1]. 
 *        This function is more of a helper methods for the bigger bSplinesWithDeriv() than a really useful
 *        function. I would adwise to just calculate everything with bSplinesWithDeriv()
 * 
 * @code
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
using namespace Eigen;

...

int kOrd = 4;
ArrayXd knots = ArrayXd::LinSpaced(11,0,1);

double x = 0.52;
int index = 5; //knots left to x;
int orderDeriv = 2;

ArrayXd splines(kOrd);
splines << 0.2,0.8,0,0; // B-splines B_(kOrd-orderDeriv, index)(x)
//could also be calculated with the bSplinesWithDeriv() method beneath

std::cout << ndxBsplines(splines, knots, index, orderDeriv) << std::endl;

//Expected output
//  20
//  40
//-140
//  80

 * @endcode
 * 
 * @param splines An array of length k (order of B-splines to calulate derivative of)
 *                that contains the correct non-zero B-splines of order k-n where n is 
 *                the degree of the derivative.
 * @param knots knot sequence of bsplines without ghost points
 * @param index the index of the knot left of the point x or the index of the knot itself
 *              (exception for the last knot, there the index i_last-1 should be given into the function)
 * @param nthDeriv order of derivative that is calculated, default = 1
 * @return ArrayXd of length k that contains the derivative values of the non-zero derivatives
 *         of splines B_k,i(x)
 */
ArrayXd ndxBsplinesHelper(ArrayXd &splines, ArrayXd &knotsInput, int index, uint nDeriv=1)
{
    int kOrd = splines.size();


    ArrayXd knots(knotsInput.size()+2*kOrd-2);
    //we add the ghost points for calculation of the splines in every point
    knots << ArrayXd::Constant(kOrd-1, knotsInput(0)),knotsInput, ArrayXd::Constant(kOrd-1, knotsInput(last));

    ArrayXd deriv = splines;

    // this loop transforms the splines array into the correct derivative
    for (int n = nDeriv; n > 0; n--)
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
 *        together with all derivatives until the nth derivative.
 *        Enter nDeriv = 0 to just calculate the splines.
 *        It also returns an index of the spline furthest to the right that is non-zero.
 *        Ghost points on both sides of the knots are automatically generated.
 *        The splines are ordered in descending order e.g. [B_i, B_i-1, B_i-2, ... , B_i-kOrd+1]
 * 
 * @code
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
using namespace Eigen;

...

int kOrd = 4;
ArrayXd knots = ArrayXd::LinSpaced(11,0,1);

double x = 0.52;
int orderDeriv = 2;
auto [mat, b] = bSplinesWithDeriv(x, knots, kOrd, orderDeriv);
std::cout << mat << std::endl;

//Expected Output
//0.00133333        0.2         20
//  0.282667        6.4         40
//  0.630667       -3.4       -140
// 0.0853333       -3.2         80
 * @endcode
 * 
 * @param x The x where to retrieve the spline values;
 * @param knotsInput the knot points without ghost points
 * @param kOrd order of the splines that are calculated
 * @param nDeriv order of higest calculated derivative, default = 0
 * @return std::tuple<MatrixXd, int> that contains the splines in column 0 and then 
 *         the n-th derivative in the nth column.
 *         The other parameter is the highest index of the non-zero splines.
 */
std::tuple<MatrixXd, int> bSplinesWithDeriv(double x, ArrayXd &knotsInput, int kOrd, int nDeriv=0)
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
            output.col(kOrd-k-1) = ndxBsplinesHelper(splines, knots, i, kOrd-k-1);
        }
    }

    output.col(0) = splines;
    return {output, i};

}


/**
 * @brief Function that returns n-th derivative of non-zero splines at position x.
 *        It also returns an index of the spline furthest to the right that is non-zero.
 *        Ghost points on both sides of the knots are automatically generated.
 *        The splines are ordered in descending order e.g. [B_i, B_i-1, B_i-2, ... , B_i-kOrd+1]
 * 
 * @code
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
using namespace Eigen;

...

int kOrd = 4;
ArrayXd knots = ArrayXd::LinSpaced(11,0,1);

double x = 0.52;
int orderDeriv = 2;
auto [arr, b] = ndxBsplines(x, knots, kOrd, orderDeriv);
std::cout << arr << std::endl;

//Expected Output
//  20
//  40
//-140
//  80
 * @endcode
 * 
 * @param x The x where to retrieve the spline values;
 * @param knotsInput the knot points without ghost points
 * @param kOrd order of the splines that are calculated
 * @param nDeriv order of higest calculated derivative, default = 1
 *               nDeriv = 0, just returns the B-splines without derivative.
 * @return std::tuple<ArrayXd, int> that contains the splines in column 0 and then 
 *         the n-th derivative in the nth column.
 *         The other parameter is the highest index of the non-zero splines.
 */
std::tuple<ArrayXd, int> ndxBsplines(double x, ArrayXd &knotsInput, int kOrd, int nDeriv=1)
{
    ArrayXd output(kOrd);
    
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

    //we iterate through to the k-2-th order in the triangular shape the problem
    //then we stop to calculate the derivative, after that we continue for 2 more iterations.
    for (int k = 1; k < kOrd-nDeriv; k++)
    {
        
        splines(k) = (knots(i+1)-x)/(knots(i+1)-knots(i-k+1))*splines(k-1);
        for (int j = 1; j < k; j++)
        {
            splines(k-j) = (x-knots(i-k+j)) / (knots(i+j)-knots(i-k+j)) * splines(k-j)
                        +(knots(i+j+1)-x) / (knots(i+j+1)-knots(i-k+j+1)) * splines(k-j-1);
        }
        splines(0) = (x-knots(i))/(knots(i+k)-knots(i))*splines(0);

    }
    output = ndxBsplinesHelper(splines, knots, i, nDeriv);
    return {output, i};

}