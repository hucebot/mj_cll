#include <mj_cll/IK.h>
#include <pyMjCll.cpp>

namespace py = pybind11;

class KinematicsMjWrapper
{
public:
    KinematicsMjWrapper(ClosedLinkageMjWrapper& clmw)
    {
        _ik = std::make_shared<mj_cll::KinematicsMj>(*clmw.getClosedLinkage().get());
    }

    bool ikLoop(py::object model, py::object data, const mj_cll::ClosedLinkageMj::VectorX& q_ref, const double lambda, const double alpha,
                const double eps, const double efc_eps, const unsigned int max_iter)
    {
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        std::uintptr_t d_raw = data.attr("_address").cast<std::uintptr_t>();


        mjModel* m_cpp = reinterpret_cast<mjModel *>(m_raw);
        mjData* d_cpp  = reinterpret_cast<mjData *>(d_raw);


       return _ik->ikLoop(*m_cpp, *d_cpp, q_ref, lambda, alpha, eps, efc_eps, max_iter);
    }

    bool ikLoopQP(py::object model, py::object data, const mj_cll::ClosedLinkageMj::VectorX& q_ref,
                  const mj_cll::ClosedLinkageMj::VectorX& qmin, const mj_cll::ClosedLinkageMj::VectorX& qmax,
                  const double lambda, const double alpha,
                  const double eps, const double efc_eps, const double reg, const unsigned int max_iter)
    {
        std::uintptr_t m_raw = model.attr("_address").cast<std::uintptr_t>();
        std::uintptr_t d_raw = data.attr("_address").cast<std::uintptr_t>();


        mjModel* m_cpp = reinterpret_cast<mjModel *>(m_raw);
        mjData* d_cpp  = reinterpret_cast<mjData *>(d_raw);

        return _ik->ikLoopQP(*m_cpp, *d_cpp, q_ref, qmin, qmax, lambda, alpha, eps, efc_eps, reg, max_iter);
    }

    void computeIKVelocity(const mj_cll::ClosedLinkageMj::VectorX& qjdot_desired, mj_cll::ClosedLinkageMj::VectorX& qadot)
    {
        _ik->computeIKVelocity(qjdot_desired, qadot);
    }

    void computeFKVelocity(const mj_cll::ClosedLinkageMj::VectorX& qdota, const mj_cll::ClosedLinkageMj::VectorX& error, mj_cll::ClosedLinkageMj::VectorX& qdotu)
    {
        _ik->computeFKVelocity(qdota, error, qdotu);
    }

    int getIterations() { return _ik->getIterations(); }

    const mj_cll::KinematicsMj::IKRESULT& getIkResult() const { return _ik->getIkResult(); }

private:
    std::shared_ptr<mj_cll::KinematicsMj> _ik;

};

PYBIND11_MODULE(pyIK, m) {
    py::class_<KinematicsMjWrapper>(m, "IKMj")
        .def(py::init<ClosedLinkageMjWrapper&>())
        .def("ikLoop", &KinematicsMjWrapper::ikLoop, py::arg("model"), py::arg("data"), py::arg("q_ref"), py::arg("lambda_"), py::arg("alpha"), py::arg("eps") = 1e-6,
             py::arg("efc_eps") = 1e-4, py::arg("max_iter") = 1000)
        .def("ikLoopQP", &KinematicsMjWrapper::ikLoopQP, py::arg("model"), py::arg("data"), py::arg("q_ref"), py::arg("qmin"), py::arg("qmax"),
             py::arg("lambda_"), py::arg("alpha"), py::arg("eps") = 1e-6, py::arg("efc_eps") = 1e-4, py::arg("reg") = 1e-9,  py::arg("max_iter") = 1000)
        .def("getIkResult", &KinematicsMjWrapper::getIkResult)
        .def("getIterations", &KinematicsMjWrapper::getIterations);

    py::enum_<mj_cll::KinematicsMj::IKRESULT>(m, "IKRESULT")
        .value("IK_NOT_CALLED", mj_cll::KinematicsMj::IKRESULT::IK_NOT_CALLED)
        .value("IK_SOLVED", mj_cll::KinematicsMj::IKRESULT::IK_SOLVED)
        .value("QP_CAN_NOT_SOLVE", mj_cll::KinematicsMj::IKRESULT::QP_CAN_NOT_SOLVE)
        .value("MAX_ITER_ACHIEVED", mj_cll::KinematicsMj::IKRESULT::MAX_ITER_ACHIEVED)
        .value("SOLUTION_STEP_NOT_INCREASE", mj_cll::KinematicsMj::IKRESULT::SOLUTION_STEP_NOT_INCREASE);

}
