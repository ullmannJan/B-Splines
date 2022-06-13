# B-Splines

A fast algorithm in C++ inspired by [De Boor's algorithm](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm) to calculate Bsplines of arbitrary order and their derivatves.

This the file ´bSplines.hpp´ contains three methods
1. ndxBsplinesHelper(ArrayXd &splines, ArrayXd &knotsInput, int index, uint nDeriv=1)
2. bSplinesWithDeriv(double x, ArrayXd &knotsInput, int kOrd, int nDeriv=0)
3. ndxBsplines(double x, ArrayXd &knotsInput, int kOrd, int nDeriv=1)
 
It is recommended to use ndxBsplines() if you just need the splines or just the derivative.
In case you need both together use method bSplinesWithDeriv()
