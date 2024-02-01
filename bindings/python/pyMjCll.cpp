#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <mj_cll/ClosedLinkage.hpp>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, int>)

class ClosedLinkageMjWrapper
{
public:
    ClosedLinkageMjWrapper(const std::string& id, const std::vector<std::string>& active_dofs, py::object model, py::object data,
                           const double row_zero_ths, const double rank_ths, const double is_similar_ths)
    {
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        std::uintptr_t d_raw = data.attr("_address").cast<std::uintptr_t>();


        _m_cpp = reinterpret_cast<mjModel *>(m_raw);
        _d_cpp  = reinterpret_cast<mjData *>(d_raw);

        _cl = std::make_shared<mj_cll::ClosedLinkageMj>(id, active_dofs, *_m_cpp, *_d_cpp, row_zero_ths, rank_ths, is_similar_ths);
    }

    void print() { _cl->print(); }

    const mj_cll::ClosedLinkageMj::MatrixX& getU() const { return _cl->getU();}

    const mj_cll::ClosedLinkageMj::MatrixX& getA() const { return _cl->getA();}

    void setP(const std::vector<std::string>& joints) { _cl->setP(joints); }

    const mj_cll::ClosedLinkageMj::MatrixX& getP() const { return _cl->getP(); }

    const std::string& getID() const { return _cl->getID(); }

    const std::vector<std::string>& getActiveDofs() const { return _cl->getActiveDofs(); }

    const std::vector<std::string>& getPassiveDofs() const { return _cl->getPassiveDofs(); }

    const unsigned int getRows() const {return _cl->getRows();}

    const unsigned int getNNZRows() const {return _cl->getNNZRows();}

    const unsigned int getCols() const {return _cl->getCols();}

    const std::vector<unsigned int>& getNNZRowsIndices() const {return _cl->getNNZRowsIndices();}

    const Eigen::Matrix<mjtNum, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& getConstraintJacobian() const {return _cl->getConstraintJacobian();}

    const mj_cll::ClosedLinkageMj::MatrixX& getPassivePartJacobian() const {return _cl->getPassivePartJacobian();}

    const mj_cll::ClosedLinkageMj::MatrixX& getActivePartJacobian() const {return _cl->getActivePartJacobian();}

    void update(bool compute_mapping_jacobian) { _cl->update(compute_mapping_jacobian); }

    const mj_cll::ClosedLinkageMj::MatrixX& getMappingJacobian() const {return _cl->getMappingJacobian();}

    const mj_cll::ClosedLinkageMj::VectorX& getConstraintError() const { return _cl->getConstraintError(); }

    const std::vector<std::string>& getJointsP() const { return _cl->getJointsP(); }

    const std::map<std::string, int>& getMujocoJointMap() const { return _cl->getMujocoJointMap(); }

    void computeMappingJacobian() { _cl->computeMappingJacobian(); }

    std::shared_ptr<mj_cll::ClosedLinkageMj> getClosedLinkage() { return _cl; }

    ~ClosedLinkageMjWrapper()
    {

    }

private:
    mjModel* _m_cpp;
    mjData*  _d_cpp;
    std::shared_ptr<mj_cll::ClosedLinkageMj> _cl;
};


PYBIND11_MODULE(pymj_cll, m) {

    py::bind_vector<std::vector<std::string>>(m, "VectorString");
    py::implicitly_convertible<py::list, std::vector<std::string>>();

    py::bind_map<std::map<std::string, int>>(m, "StringIntMap");
    py::implicitly_convertible<py::dict, std::map<std::string, int>>();


    py::class_<ClosedLinkageMjWrapper>(m, "ClosedLinkageMjWrapper")
        .def(py::init<const std::string &, const std::vector<std::string> &, py::object, py::object, const double, const double, const double>(),
             py::arg("id"), py::arg("active_dofs"), py::arg("model"), py::arg("data"), py::arg("row_zero_ths") = 1e-5, py::arg("rank_ths") = 1e-5, py::arg("is_similar_ths") = 1e-4)
        .def("print", &ClosedLinkageMjWrapper::print)
        .def("getU", &ClosedLinkageMjWrapper::getU, py::return_value_policy::reference_internal)
        .def("getA", &ClosedLinkageMjWrapper::getA, py::return_value_policy::reference_internal)
        .def("setP", &ClosedLinkageMjWrapper::setP)
        .def("getP", &ClosedLinkageMjWrapper::getP, py::return_value_policy::reference_internal)
        .def("getID", &ClosedLinkageMjWrapper::getID, py::return_value_policy::reference_internal)
        .def("getActiveDofs", &ClosedLinkageMjWrapper::getActiveDofs, py::return_value_policy::reference_internal)
        .def("getPassiveDofs", &ClosedLinkageMjWrapper::getPassiveDofs, py::return_value_policy::reference_internal)
        .def("getRows", &ClosedLinkageMjWrapper::getRows)
        .def("getNNZRows", &ClosedLinkageMjWrapper::getNNZRows)
        .def("getCols", &ClosedLinkageMjWrapper::getCols)
        .def("getNNZRowsIndices", &ClosedLinkageMjWrapper::getNNZRowsIndices, py::return_value_policy::reference_internal)
        .def("getConstraintJacobian", &ClosedLinkageMjWrapper::getConstraintJacobian, py::return_value_policy::reference_internal)
        .def("getPassivePartJacobian", &ClosedLinkageMjWrapper::getPassivePartJacobian, py::return_value_policy::reference_internal)
        .def("getActivePartJacobian", &ClosedLinkageMjWrapper::getActivePartJacobian, py::return_value_policy::reference_internal)
        .def("update", &ClosedLinkageMjWrapper::update, py::arg("compute_mapping_jacobian") = true)
        .def("getMappingJacobian", &ClosedLinkageMjWrapper::getMappingJacobian, py::return_value_policy::reference_internal)
        .def("getJointsP", &ClosedLinkageMjWrapper::getJointsP, py::return_value_policy::reference_internal)
        .def("getConstraintError", &ClosedLinkageMjWrapper::getConstraintError, py::return_value_policy::reference_internal)
        .def("computeMappingJacobian", &ClosedLinkageMjWrapper::computeMappingJacobian)
        .def("getMujocoJointMap", &ClosedLinkageMjWrapper::getMujocoJointMap, py::return_value_policy::reference_internal);


}
