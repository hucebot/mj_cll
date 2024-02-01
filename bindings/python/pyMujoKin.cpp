#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <mj_cll/MujoKin.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

class MujoKinWrapper
{
public:
    MujoKinWrapper(){}

    ~MujoKinWrapper(){}

    void createScene(py::object model)
    {
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        mjModel* m_cpp = reinterpret_cast<mjModel *>(m_raw);

        _mjk.createScene(m_cpp);
    }

    void show(py::object model, py::object data)
    {
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        std::uintptr_t d_raw = data.attr("_address").cast<std::uintptr_t>();

        mjModel* m_cpp = reinterpret_cast<mjModel *>(m_raw);
        mjData* d_cpp  = reinterpret_cast<mjData *>(d_raw);

        _mjk.show(m_cpp, d_cpp);
    }

    void loop(py::object model, py::object data, std::function<void()> func, const int us, const int stop_in)
    {
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        std::uintptr_t d_raw = data.attr("_address").cast<std::uintptr_t>();

        mjModel* m_cpp = reinterpret_cast<mjModel *>(m_raw);
        mjData* d_cpp  = reinterpret_cast<mjData *>(d_raw);

        _mjk.loop(m_cpp, d_cpp, func, us, std::chrono::milliseconds(stop_in));
    }

private:
    mj_cll::MujoKin _mjk;

};


PYBIND11_MODULE(pymujokin, m) {

    py::class_<MujoKinWrapper>(m, "MujoKinWrapper")
        .def(py::init<>())
        .def("createScene", &MujoKinWrapper::createScene, py::arg("model"))
        .def("show", &MujoKinWrapper::show, py::arg("model"), py::arg("data"))
        .def("loop", &MujoKinWrapper::loop, py::arg("model"), py::arg("data"), py::arg("func"), py::arg("us") = 0.0, py::arg("stop_in") = -1);

}
