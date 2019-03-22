#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace std;
using namespace Eigen;

/*
Test Example: f(v)=x^2+(x+y-10)^2+(2*y-z)^2+2*z;
Approximate Hessian Matrix using SR1.
*/

double f(Vector3d v)
{
	double x = v(0);
	double y = v(1);
	double z = v(2);
	double res;

	res = x * x + (x + y - 10)*(x + y - 10) + (2 * y - z)*(2 * y - z) + 2 * z;

	return res;
}

Vector3d J(Vector3d v)
{
	Vector3d J;
	double x = v(0);
	double y = v(1);
	double z = v(2);

	double dx = 2 * x + 2 * (x + y - 10);
	double dy = 2 * (x + y - 10) + 2 * (2 * y - z) * 2;
	double dz = 2 * (z - 2 * y) + 2;

	J(0) = dx; J(1) = dy; J(2) = dz;
	return J;
}

bool PositiveDefinite(Matrix3d G)
{
	Vector3cd E = G.eigenvalues();
	
	bool res = E(0).real() > 0 && E(1).real() > 0 && E(2).real() > 0;

	return res;
}

int main()
{
	Vector3d v(100, 100, 100);
	double u = 0.25;
	Matrix3d G; G.setIdentity();
	Matrix3d I; I.setIdentity();
	
	while (1)
	{
		Vector3d g = J(v);
		if (g.norm() < 1e-6) break;
		while (!PositiveDefinite(G + u * I))
		{
			u *= 4;
		}
		Matrix3d K = G + u * I;
		Vector3d s = -K.inverse()*g;

		double fn = f(v + s);
		double qk = f(v) + g.transpose()*s + 0.5*s.transpose()*G*s;
		double rk = (f(v) - fn) / (f(v)-qk);

		if (rk < 0.25) u *= 4.0;
		else if (rk > 0.75) u /= 2.0;

		if (rk > 0)
		{
			v = v + s;
			Vector3d sk = -s;
			Vector3d yk = g-J(v);
			double den = (sk - G * yk).transpose()*yk;
			G = G + (sk - G * yk)*(sk - G * yk).transpose() / den;
		}
	}

	cout << v.transpose() << endl;
	cout << f(v) << endl;

}