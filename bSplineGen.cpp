#pragma once
#include <iostream>
#include <tuple>
#include <Eigen/Dense>
#include "saveToFile.cpp"

using namespace Eigen;

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
std::tuple<MatrixXd, int> calcBSplineAndDeriv(double x, ArrayXd &knotsInput, int kOrd)
{
    MatrixXd output = MatrixXd::Zero(kOrd,3);
 
    //the last value hold the knot of the point, so that we have a reference to our splines
    ArrayXd splines = ArrayXd::Zero(kOrd);
        
    //for calculation of derivative
    ArrayXd deriv1 = ArrayXd::Zero(kOrd);
    ArrayXd deriv2 = ArrayXd::Zero(kOrd);

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
    //std::cout << "i-start: " << i << " knots(i): " << knots(i) << std::endl;

    //we iterate through to the k-2-th order in the triangular shape the problem
    //then we stop to calculate the derivative, after that we continue for 2 more iterations.
    for (int k = 1; k < kOrd-2; k++)
    {
        splines(k) = (knots(i+1)-x)/(knots(i+1)-knots(i-k+1))*splines(k-1);
        for (int j = 1; j < k; j++)
        {
            splines(k-j) = (x-knots(i-k+j)) / (knots(i+j)-knots(i-k+j)) * splines(k-j)
                        +(knots(i+j+1)-x) / (knots(i+j+1)-knots(i-k+j+1)) * splines(k-j-1);
        }
        splines(0) = (x-knots(i))/(knots(i+k)-knots(i))*splines(0);
    }

    //  =================== now we calculate the 2nd derivative
    // we will calculate the border cases manually

    int k = kOrd-1;
    //std::cout << "splines of k-2\n" << splines << std::endl;
    
    deriv2(0) =  (k-1)*(k)*splines(0) /(knots(i+k)-knots(i)) /(knots(i+k-1)-knots(i));
    deriv2(1) = (k-1)*(k)*(     + splines(1) /(knots(i-1+k)-knots(i-1)) /(knots(i-1+k-1)-knots(i-1))
                                - splines(0) /(knots(i-1+k)-knots(i-1)) /(knots(i-1+k)-knots(i))
                                - splines(0) /(knots(i-1+k+1)-knots(i)) /(knots(i-1+k)-knots(i)));

    for (int j = 0; j < k-3; j++)
    {
        deriv2(j) = (k-1)*(k)*(
                                + splines(j) /(knots(i-j+k)-knots(i-j)) /(knots(i-j+k-1)-knots(i-j))
                                - splines(j-1) /(knots(i-j+k)-knots(i-j)) /(knots(i-j+k)-knots(i-j+1))
                                - splines(j-1) /(knots(i-j+k+1)-knots(i-j+1)) /(knots(i-j+k)-knots(i-j+1))
                                + splines(j-2) /(knots(i-j+k+1)-knots(i-j+1)) /(knots(i-j+k+1)-knots(i-j+2))
                                );
    }

    //std::cout << "k: " << k << " value: " <<  (knots(i+k))  << std::endl; 
    deriv2(k-1)  = (k-1)*(k)*(  - splines(k-2) /(knots(i+1)-knots(i-k+1)) /(knots(i+1)-knots(i-k+2))
                                - splines(k-2) /(knots(i+2)-knots(i-k+2)) /(knots(i+1)-knots(i-k+2))
                                + splines(k-3) /(knots(i+2)-knots(i-k+2)) /(knots(i+2)-knots(i-k+3)));
    deriv2(k) = (k-1)*(k) * (splines(k-2) /(knots(i+1)-knots(i-k+1)) /(knots(i+1)-knots(i-k+2)));

    // =============================================================================
    // continue calculating the splines
    k = kOrd-2;
    splines(k) = (knots(i+1)-x)/(knots(i+1)-knots(i-k+1))*splines(k-1);
    for (int j = 1; j < k; j++)
    {
        splines(k-j) = (x-knots(i-k+j)) / (knots(i+j)-knots(i-k+j)) * splines(k-j)
                    +(knots(i+j+1)-x) / (knots(i+j+1)-knots(i-k+j+1)) * splines(k-j-1);
    }
    splines(0) = (x-knots(i))/(knots(i+k)-knots(i))*splines(0);
    
    // =============================================================================
    // calculate the first derivative
    deriv1(0) = (kOrd-1)*splines(0) /(knots(i+kOrd-1)-knots(i));
    for (int j = 1; j < kOrd-1; j++)
    {
        deriv1(j) = (kOrd-1)*splines(j) / (knots(i+kOrd-1-j)-knots(i-j))
                    - (kOrd-1)*splines(j-1) / (knots(i+kOrd-j)-knots(i-j+1));
    }
    deriv1(kOrd-1) = -(kOrd-1)*splines(kOrd-2) / (knots(i+1)-knots(i-kOrd+2));

    // =============================================================================
    // and the last iteration of normal splines
    k = kOrd-1;
    splines(k) = (knots(i+1)-x)/(knots(i+1)-knots(i-k+1))*splines(k-1);
    for (int j = 1; j < k; j++)
    {
        splines(k-j) = (x-knots(i-k+j)) / (knots(i+j)-knots(i-k+j)) * splines(k-j)
                    +(knots(i+j+1)-x) / (knots(i+j+1)-knots(i-k+j+1)) * splines(k-j-1);
    }
    splines(0) = (x-knots(i))/(knots(i+k)-knots(i))*splines(0);

    output.col(0) = splines;
    output.col(1) = deriv1;
    output.col(2) = deriv2;
    return {output, i};
}

void testCalcBSplineAndDeriv(double xx, ArrayXd knots)
{
    auto [test, idx] = calcBSplineAndDeriv(xx, knots, 4);
    std::cout << "the setup went through" << std::endl;
    std::cout << test  << std::endl;
    std::cout << "test if sum is one: " << test.col(0).sum()  << std::endl;
}

/**
 * @brief Function that returns spline values at position x and only those that are not zero
 * 
 * @param x x value, where spline value is retrieved
 * @param knotsInput the knot points that have no ghost points yet
 * @param kOrd order of the splines that are calculated
 * @return ArrayXd that contains the values of the spline and the index i in the ghosted array
 *         remember that the splines are ordered as i, i-1, i-2, ... i-kOrd+1 
 */
ArrayXd calcBSpline(double x, ArrayXd &knotsInput, int kOrd)
{
    //the last value hold the knot of the point, so that we have a reference to our splines
    ArrayXd splines = ArrayXd::Zero(kOrd+1);
    ArrayXd knots(knotsInput.size()+2*kOrd-2);

    //we add the ghost points for calculation of the splines in every point
    knots << ArrayXd::Constant(kOrd-1, knotsInput(0)),knotsInput, ArrayXd::Constant(kOrd-1, knotsInput(last));

    // we set the lowest layer BSplines manually
    splines(0) = 1;

    int i = kOrd-1; //we start at the first non ghost point

    //if x == knot we will select that knot and return instantly
    if (x == knotsInput(last))
    {
        //special case is granted to the end ghost points as we want the first one here
        //we want the index where the last element starts, so no ghost points but the last real element
        i = knotsInput.size()-1+i;
        //The splines all become zero except for the last one, so we can return it manually
        splines(splines.size()-1) = i;
        return splines;
    }
    
    //we search for the knot left of the x-value
    while (x >= knots(i))
    {
        i++;
    }
    i--;
    //std::cout << "i-start: " << i << " knots(i): " << knots(i) << std::endl;

    //we iterate through to the k-th order in the triangular shape the problem consists of
    for (int k = 1; k < kOrd; k++)
    {
        splines(k) = (knots(i+1)-x)/(knots(i+1)-knots(i-k+1))*splines(k-1);
        for (int j = 1; j < k; j++)
        {
            splines(k-j) = (x-knots(i-k+j)) / (knots(i+j)-knots(i-k+j)) * splines(k-j)
                        +(knots(i+j+1)-x) / (knots(i+j+1)-knots(i-k+j+1)) * splines(k-j-1);
        }
        splines(0) = (x-knots(i))/(knots(i+k)-knots(i))*splines(0);
       
        //std::cout << "================" << std::endl;
        //std::cout << splines << std::endl;
    }

    splines(splines.size()-1) = i;

    return splines;
}

/**
 * @brief Function that writes a file with given points according to the splines
 * 
 * @param xSpace x values where to sample the splines
 * @param knots the knots to construct the splines
 * @param kOrd the order of the splines
 */
void splines(ArrayXd &xSpace, ArrayXd &knots, int kOrd)
{
    for (int i = 0; i < 3; i++)
    {
        MatrixXd output = MatrixXd::Zero(xSpace.size(), knots.size()+kOrd-1);
        output.col(0) = xSpace;
        for (int j = 0; j < xSpace.size(); j ++)
        {
            auto [input, index] = calcBSplineAndDeriv(xSpace(j),knots, kOrd);
            ArrayXd spline = input.col(i);
            //std::cout << j << ". j =======\n" << input << std::endl;
            for (int i = 0; i < kOrd; i++)
            {
                output(j, index-i+1) = spline(i);
            }
        }
        std::string s = "data/splinesD" + std::to_string(i) + ".csv";
        saveFile(output,s);
    }
}

ArrayXd splinesCoeffMult(ArrayXd &xSpace, VectorXd &coeffsVec, ArrayXd &knots, int kOrd)
{

        ArrayXd coeffs (2+coeffsVec.size());
        coeffs << 0, coeffsVec, 0;
        //std::cout << coeffs;
        ArrayXd output = ArrayXd::Zero(xSpace.size());
        for (int j = 0; j < xSpace.size(); j++)
        {
            auto [mat, idx] = calcBSplineAndDeriv(xSpace(j),knots, kOrd);
            ArrayXd spline = mat.col(0);
            
            int index = idx-kOrd+1;
            //c1 is missing so we will have the second spline 
            double sum = 0;
            for (int i = 0; i < kOrd; i++)
            {
                //std::cout << "index" << index << "i" << coeffs.size() << std::endl;
                sum += coeffs(index+i)*spline(kOrd-i-1);
            }
            
            output(j) = sum;
        }
        return output;
}
