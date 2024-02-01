#ifndef _MJ_CLL_IK_H_
#define _MJ_CLL_IK_H_

#include <mj_cll/ClosedLinkage.hpp>
#include <mj_cll/eiQuadProg.hpp>

namespace mj_cll
{
    template <class T, class M, class D>
    class Kinematics
    {
        public:
            typedef std::shared_ptr<Kinematics> Ptr;

            enum class IKRESULT
            {
                IK_NOT_CALLED,
                IK_SOLVED,
                QP_CAN_NOT_SOLVE,
                MAX_ITER_ACHIEVED,
                SOLUTION_STEP_NOT_INCREASE
            };

            Kinematics(ClosedLinkage<T, M, D>& closed_linkage):
                _closed_linkage(closed_linkage),
                _iter(0),
                _result(IKRESULT::IK_NOT_CALLED)
            {

            }

            virtual ~Kinematics(){}

            /**
             * @brief computeIKVelocity computes the velocity of the actuated dofs given desired values for a portion of the passive ones
             * @param qjdot_desired desired passive joint velocities
             * @param qadot actuated joint velocities
             */
            void computeIKVelocity(const typename ClosedLinkage<T, M, D>::VectorX& qjdot_desired,
                                   typename ClosedLinkage<T, M, D>::VectorX& qadot)
            {
                if (qjdot_desired.size() != _closed_linkage.getPassiveDofs().size())
                  std::runtime_error("qjdot_desired.size() != _closed_linkage.getPassiveDofs().size()");

                //  qadot = (_P * _closed_linkage.getPassivePartJacobian().householderQr().solve(
                //      -_closed_linkage.getActivePartJacobian())).householderQr().solve(qjdot_desired);

                this->solve(_closed_linkage.getP() * _closed_linkage.getPassivePartJacobian().householderQr().solve(
                                -_closed_linkage.getActivePartJacobian()), qjdot_desired, qadot);
            }

            /**
             * @brief computeFKVelocity given the velocities of the actuated joints, computes the velocity of the passive joints taking into account
             * constraint error (better to be used in an IK loop)
             * @param qdota actuated joint velocities
             * @param error constraint error
             * @param qdotu passive joint velocities
             */
            void computeFKVelocity(const typename ClosedLinkage<T, M, D>::VectorX& qdota,
                                   const typename ClosedLinkage<T, M, D>::VectorX& error,
                                   typename ClosedLinkage<T, M, D>::VectorX& qdotu)
            {
                //  qdotu = _closed_linkage.getPassivePartJacobian().householderQr().solve(
                //      -_closed_linkage.getActivePartJacobian() * qdota - error);
                this->solve(_closed_linkage.getPassivePartJacobian(), -_closed_linkage.getActivePartJacobian() * qdota - error, qdotu);
            }

            /**
             * @brief ikLoop resolve the ik problem meaning to find the values of all the passive and active dofs position given the desired value for
             * a part of the passive ones.
             * @param model mujoco model
             * @param data mujoco data
             * @param q_ref position reference for a subset of the passive dofs (selected using the setP in the closed linkage)
             * @param lambda positive gain which multiplies the error between the desired values of the passive joints and the actual one
             * @param alpha positive gain that multiplies the error of the closed linkage constraints
             * @param eps threshold used to stop the iterations, it is used to check the error on the reference
             * @param efc_eps threshold used to stop the iterations, it is used to check the error on the closed linkage constraint
             * @param max_iter maximum number of iterations allowed for the IK
             * @return true (IK_SOLVED, SOLUTION_STEP_NOT_INCREASE), false (MAX_ITER_ACHIEVED):
             * - IK_SOLVED: error norm <= eps and constraint error norm <= efc_eps
             * - SOLUTION_STEP_NOT_INCREASE: dq step <= 1e-12 and constraint error norm <= efc_eps
             * - MAX_ITER_ACHIEVED: iter > max_iter
             */
            bool ikLoop(M& model, D& data, const typename ClosedLinkage<T, M, D>::VectorX& q_ref, const double lambda, const double alpha,
                        const double eps = 1e-6, const double efc_eps = 1e-4, const unsigned int max_iter = 1000)
            {
                _q = Eigen::Map<typename ClosedLinkage<T, M, D>::VectorX, Eigen::Unaligned>(data.qpos, model.nq);
                _dq.setZero(_q.size());

                _iter = 0;
                while(true)
                {
                    std::memcpy(data.qpos, _q.data(), _q.size() * sizeof(_q.data()[0]));

                    mj_fwdPosition(&model, &data);


                    _closed_linkage.update(false);


                    _e = q_ref - _closed_linkage.getP() * (_closed_linkage.getU().transpose()*_q);


                    this->computeIKVelocity(lambda * _e, _dqa);
                    this->computeFKVelocity(_dqa, alpha*_closed_linkage.getConstraintError(), _dqu);

                    _dq = _closed_linkage.getA() * _dqa + _closed_linkage.getU() * _dqu;

                    if(_dq.norm() < 1e-12 && _closed_linkage.getConstraintError().norm() <= efc_eps)
                    {
                        _result = IKRESULT::SOLUTION_STEP_NOT_INCREASE;
                        return true;
                    }

                    _q += _dq;

                    if(_e.norm() <= eps && _closed_linkage.getConstraintError().norm() <= efc_eps)
                    {
                        _result = IKRESULT::IK_SOLVED;
                        return true;
                    }


                    _iter++;
                    if(_iter > max_iter)
                    {
                        _result = IKRESULT::MAX_ITER_ACHIEVED;
                        return false;
                    }                    
                }
            }

            /**
             * @brief computeVelocity of the whole closed kinematic chain given desired reference for a sub-section of the passive dofs and taking into account joint limits
             * @param model mujoco model
             * @param data mujoco data
             * @param qjdot_desired desired passive joint velocities
             * @param error constraint error
             * @param qmin joint limits
             * @param qmax joint limits
             * @param dt loop time
             * @param qdot passive and active velocities
             * @param reg regularization term used in the QP
             * @return false if QP returns error
             */
            bool computeIFKVelocity(M& model, D& data, const typename ClosedLinkage<T, M, D>::VectorX& qjdot_desired,
                                 const typename ClosedLinkage<T, M, D>::VectorX& error,
                                 const typename ClosedLinkage<T, M, D>::VectorX& qmin, const typename ClosedLinkage<T, M, D>::VectorX& qmax,
                                 const double dt,
                                 typename ClosedLinkage<T, M, D>::VectorX& qdot, const double reg = 1e-9)
            {
                _q = Eigen::Map<typename ClosedLinkage<T, M, D>::VectorX, Eigen::Unaligned>(data.qpos, model.nq);

                _A.setZero(_closed_linkage.getP().rows() + _closed_linkage.getConstraintJacobian().rows(), _q.size());
                _e.setZero(_A.rows());

                _C.setZero(2*_q.size(), _q.size());
                _C.block(0,0,_q.size(),_q.size()).setIdentity();
                _C.block(_q.size(),0,_q.size(),_q.size()) = -_C.block(0,0,_q.size(),_q.size());
                _c.setZero(2*_q.size());

                _A.block(0,0,_closed_linkage.getP().rows(), _q.size()) = _closed_linkage.getP() * _closed_linkage.getU().transpose();
                _A.block(_closed_linkage.getP().rows(),0, _closed_linkage.getConstraintJacobian().rows(), _q.size()) = _closed_linkage.getConstraintJacobian();

                _e.segment(0, _closed_linkage.getP().rows()) = qjdot_desired;
                _e.segment(_closed_linkage.getP().rows(), _closed_linkage.getConstraintJacobian().rows()) = -error();

                computeJointLimitsConstraint(qmin, qmax, _q, dt);

                if(!solveQP(_A, _e, _C.transpose(), _c, reg, qdot))
                    return false;
                return true;
            }

            /**
             * @brief ikLoopQP resolve the ik problem meaning to find the values of all the passive and active dofs position given the desired value for
             * a part of the passive ones. In this version a QP solver is used therefore we can specify also linear constraints, for now just joint limits.
             * @param model mujoco model
             * @param data mujoco data
             * @param q_ref position reference for a subset of the passive dofs (selected using the setP in the closed linkage)
             * @param qmin joint limits
             * @param qmax joint limits
             * @param lambda positive gain which multiplies the error between the desired values of the passive joints and the actual one
             * @param alpha positive gain that multiplies the error of the closed linkage constraints
             * @param eps threshold used to stop the iterations, it is used to check the error on the reference
             * @param efc_eps threshold used to stop the iterations, it is used to check the error on the closed linkage constraint
             * @param reg regularization term used in the QP
             * @param max_iter maximum number of iterations allowed for the IK
             * @return true (IK_SOLVED, SOLUTION_STEP_NOT_INCREASE), false (QP_CAN_NOT SOLVE, MAX_ITER_ACHIEVED)
             * - IK_SOLVED: error norm <= eps and constraint error norm <= efc_eps
             * - SOLUTION_STEP_NOT_INCREASE: dq step <= 1e-12 and constraint error norm <= efc_eps
             * - MAX_ITER_ACHIEVED: iter > max_iter
             * - QP_CAN_NOT_SOLVE: internal QP error
             */
            virtual bool ikLoopQP(M& model, D& data, const typename ClosedLinkage<T, M, D>::VectorX& q_ref,
                          const typename ClosedLinkage<T, M, D>::VectorX& qmin, const typename ClosedLinkage<T, M, D>::VectorX& qmax,
                          const double lambda, const double alpha,
                          const double eps = 1e-6, const double efc_eps = 1e-4, const double reg = 1e-9, const unsigned int max_iter = 1000)
            {
                _q = Eigen::Map<typename ClosedLinkage<T, M, D>::VectorX, Eigen::Unaligned>(data.qpos, model.nq);
                _dq.setZero(_q.size());

                _A.setZero(_closed_linkage.getP().rows() + _closed_linkage.getConstraintJacobian().rows(), _q.size());
                _e.setZero(_A.rows());

                _C.setZero(2*_q.size(), _q.size());
                _C.block(0,0,_q.size(),_q.size()).setIdentity();
                _C.block(_q.size(),0,_q.size(),_q.size()) = -_C.block(0,0,_q.size(),_q.size());
                _c.setZero(2*_q.size());

                _iter = 0;
                while(true)
                {
                    std::memcpy(data.qpos, _q.data(), _q.size() * sizeof(_q.data()[0]));

                    mj_fwdPosition(&model, &data);

                    _closed_linkage.update(false);

                    _A.block(0,0,_closed_linkage.getP().rows(), _q.size()) = _closed_linkage.getP() * _closed_linkage.getU().transpose();
                    _A.block(_closed_linkage.getP().rows(),0, _closed_linkage.getConstraintJacobian().rows(), _q.size()) = _closed_linkage.getConstraintJacobian();

                    _e.segment(0, _closed_linkage.getP().rows()) = lambda * (q_ref - _closed_linkage.getP() * (_closed_linkage.getU().transpose()*_q));
                    _e.segment(_closed_linkage.getP().rows(), _closed_linkage.getConstraintJacobian().rows()) = -alpha * _closed_linkage.getConstraintError();

                    computeJointLimitsConstraint(qmin, qmax, _q, 1.);


                    if(!solveQP(_A, _e, _C.transpose(), _c, reg, _dq))
                    {
                        _result = IKRESULT::QP_CAN_NOT_SOLVE;
                        return false;
                    }

                    _q += _dq;

                    if(_e.segment(0, _closed_linkage.getP().rows()).norm() <= eps && _closed_linkage.getConstraintError().norm() <= efc_eps)
                    {
                        _result = IKRESULT::IK_SOLVED;
                        return true;
                    }

                    if(_dq.norm() < 1e-12 && _closed_linkage.getConstraintError().norm() <= efc_eps)
                    {
                        _result = IKRESULT::SOLUTION_STEP_NOT_INCREASE;
                        return true;
                    }

                    _iter++;
                    if(_iter > max_iter)
                    {
                        _result = IKRESULT::MAX_ITER_ACHIEVED;
                        return false;
                    }

                }
            }

            /**
             * @brief getIterations return number of iterations after perfoming IK
             * @return number of iterations
             */
            int getIterations() { return _iter; }

            /**
             * @brief getIkResult return IK result after performing IK
             * @return ik result
             */
            const IKRESULT& getIkResult() const { return _result; }


        private:
            ClosedLinkage<T, M, D>& _closed_linkage;

            typename ClosedLinkage<T, M, D>::VectorX _q, _e, _dqa, _dqu, _dq;

            typename ClosedLinkage<T, M, D>::MatrixX _H, _A, _C;
            typename ClosedLinkage<T, M, D>::VectorX _g, _c;
            Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,Eigen::Lower> _chol;

            IKRESULT _result;

            void solve(const typename ClosedLinkage<T, M, D>::MatrixX& A, const typename ClosedLinkage<T, M, D>::VectorX& b,  typename ClosedLinkage<T, M, D>::VectorX& x)
            {
                _H = A.transpose()*A;
                _g = A.transpose()*b;
                x = _H.llt().solve(_g);
            }

            bool solveQP(const typename ClosedLinkage<T, M, D>::MatrixX& A, const typename ClosedLinkage<T, M, D>::VectorX& b,
                         const typename ClosedLinkage<T, M, D>::MatrixX& Cineq, const typename ClosedLinkage<T, M, D>::VectorX& cineq,
                         const double eps,
                         typename ClosedLinkage<T, M, D>::VectorX& x)
            {
                _H = A.transpose()*A;
                _g = -A.transpose()*b;


                for(unsigned int i = 0; i < _H.rows(); ++i)
                    _H(i, i) += eps;


                _chol = _H.llt();
                double f_value = solve_quadprog2(_chol, _H.trace(), _g,
                                                typename ClosedLinkage<T, M, D>::MatrixX(), typename ClosedLinkage<T, M, D>::VectorX(),
                                                Cineq, cineq,
                                                x);
                if(f_value == std::numeric_limits<double>::infinity())
                {
                    return false;
                }
                return true;
            }

            void computeJointLimitsConstraint(const typename ClosedLinkage<T, M, D>::VectorX& qmin, const typename ClosedLinkage<T, M, D>::VectorX& qmax, typename ClosedLinkage<T, M, D>::VectorX& q,
                                              const double dt)
            {
                _c.segment(0, qmin.size()) = -(qmin-q)/dt;
                _c.segment(qmin.size(), qmax.size()) = (qmax-q)/dt;
            }

            unsigned int _iter;
    };





typedef Kinematics<mjtNum, mjModel, mjData> KinematicsMj;
}

#endif
