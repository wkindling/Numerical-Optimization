#include <Eigen/Dense>
#include <iostream>
#include <cmath>

/*
Test Example of LM trust region method. 
v=(x,y,z); 
f(v)=x^2+(x+y-10)^2+(2*y-z)^2+2*z;

The user has to input the Jacobian Matrix of objective function, and approximate Hessian Matrix using quasi-Newton Method.
*/
using namespace std;
using namespace Eigen;

double f(Vector3d v)
{
	double x = v.x();
	double y = v.y();
	double z = v.z();

	double res = x * x + (x + y - 10)*(x + y - 10) + (2 * y - z)*(2 * y - z)+2*z;
	return res;
}

Vector3d J(Vector3d v)
{
	double x = v.x();
	double y = v.y();
	double z = v.z();

	double dx = 2 * x + 2 * (x + y - 10);
	double dy = 2 * (x + y - 10) + 2 * (2 * y - z) * 2;
	double dz = 2 * (z - 2 * y)+2;

	Vector3d res(dx, dy, dz);
	return res;
}

bool PositiveDefinite(Matrix3d M)
{
	Vector3cd E = M.eigenvalues();

	bool res = E(0).real() >0 && E(1).real() >0 && E(2).real() >0;
	return res;
}

int main()
{
	Vector3d v(10000, 10000, 10000);
	Vector3d g = J(v);
	Matrix3d G = g * g.transpose();
	Matrix3d I;
	I.setIdentity();
	double u = 0.25;

	while (1)
	{
		Vector3d new_g = J(v);
		if (new_g.norm() < 0.0001) break;
		while (!PositiveDefinite(G + u * I))
		{
			u = u * 4;
		}
		Matrix3d K = G + u * I;
		Vector3d s = -K.inverse()*new_g;

		double fn = f(v + s);
		double qk = f(v) + g.transpose()*s + 0.5*s.transpose()*G*s;
		double rk = (f(v) - fn) / (f(v) - qk);
		if (rk < 0.25) u = 4 * u;
		else if (rk > 0.75) u = u / 2.0;
		if (rk > 0) v = v + s;

		Vector3d is(1.0 / s(0), 1.0 / s(1), 1.0 / s(2));
		G = (new_g - g)*is.transpose();
		g = new_g;
	}
	cout << f(v) << endl;
	cout << v.transpose() << endl;
}