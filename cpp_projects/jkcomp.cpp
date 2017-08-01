#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>

namespace py = pybind11;

std::vector<py::array> form_JK(py::array_t<double> I, py::array_t<double> D)
{
	py::buffer_info I_info = I.request();
	py::buffer_info D_info = D.request();

	if(I_info.ndim != 4)
		throw std::runtime_error("I is not a rank-4 tensor");
	if(D_info.ndim != 2)
		throw std::runtime_error("D is not a matrix");

    size_t dim = D_info.shape[0];
    size_t dim2 = dim*dim;
    size_t dim3 = dim*dim2;
	const double * I_data = static_cast<double *>(I_info.ptr);
	const double * D_data = static_cast<double *>(D_info.ptr);
	
	std::vector<double> J_data(dim * dim);
	std::vector<double> K_data(dim * dim);

	for(size_t p = 0; p < dim; p++)
	{
		for(size_t q = 0; q <= p; q++)
		{
			double Jvalue = 0.0;
			double Kvalue = 0.0;
			for(size_t r = 0; r < dim; r++)
			{
				for(size_t s = 0; s < r; s++)
				{
					Jvalue += 2.0*I_data[p*dim3 + q*dim2 + r*dim + s] * D_data[r*dim + s];
				}
				Jvalue += I_data[p*dim3 + q*dim2 + r*dim + r] * D_data[r*dim + r];
				for(size_t i = 0; i < dim; i++)
				{
					Kvalue += I_data[p*dim3 + r*dim2 + q*dim + i] * D_data[r*dim + i];
				}
			}
            J_data[p * dim + q] = Jvalue;
	        J_data[q * dim + p] = Jvalue;
			K_data[p * dim + q]	= Kvalue;
			K_data[q * dim + p]	= Kvalue;

		}
	}
	py::buffer_info Jbuf =
	{
		J_data.data(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		2,
		{dim, dim},
		{dim * sizeof(double), sizeof(double)}
	};

	py::buffer_info Kbuf =
	{
		K_data.data(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		2,
		{dim, dim},
		{dim * sizeof(double), sizeof(double)}
	};

	
	
	py::array J(Jbuf);
    py::array K(Kbuf);
    return {J, K};
	//return py::make_tuple(Jbuf, Kbuf);
}


PYBIND11_PLUGIN(jkcomp)
{
	py::module m("jkcomp", "computes J and K");

	m.def("form_JK", &form_JK, "Computes J and K");

	return m.ptr();
}
