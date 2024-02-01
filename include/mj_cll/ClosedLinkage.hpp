#ifndef _MJ_CLL_CLOSED_LINKAGE_H_
#define _MJ_CLL_CLOSED_LINKAGE_H_

#include <mujoco/mujoco.h>
#include <memory>
#include <Eigen/Dense>
#include <mj_cll/Utils.h>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>


namespace mj_cll{

template <class T, class M, class D>
class ClosedLinkage
{
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorX;

    typedef std::shared_ptr<ClosedLinkage<T, M, D>> Ptr;

    /**
     * @brief ClosedLinkage constructor
     * @param id a name of the closed linkage
     * @param active_dofs vector of dof names representing the "actuated" dofs in the linkage
     * @param model a mujoco model
     * @param data mujoco data structure
     */
    ClosedLinkage(const std::string& id, const std::vector<std::string>& active_dofs, M& model, D& data,
                  const double row_zero_ths = 1e-5, const double rank_ths = 1e-5, const double is_similar_ths = 1e-4):
    _id(id),
    _va(active_dofs),
    _model(model),
    _data(data)
    {
        _cols = _model.nq;
        _rows = _data.ne;

        _efc_J.setZero(_rows, _cols);
        std::vector<unsigned int> tmp;
        for(unsigned int i = 0; i < _rows; ++i)
            tmp.push_back(i);
        copy_rows(_data.efc_J, _rows, _cols, tmp, _efc_J);

        try
        {
            compute_constraints_rows_indices(_efc_J, _nnz_rows, _efc_J_rank, row_zero_ths, rank_ths, is_similar_ths);
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what();
            std::cerr<<"This error is now threated as a warning!"<<std::endl;
        }

        _J.setZero(_nnz_rows.size(), _cols);
        _ce.setZero(_nnz_rows.size());

        mj_cll::get_joint_names(&_model, _vu);

        ///@todo: handle case user input error for active_dofs!
        for(unsigned int i = 0; i < _va.size(); i++)
           _vu.erase(std::remove(_vu.begin(), _vu.end(), _va[i]), _vu.end());

        for(auto& adof : _va)
            _map_mujoco_joints[adof] = mj_name2id(&_model, mjOBJ_JOINT, adof.c_str());
        for(auto& udof : _vu)
            _map_mujoco_joints[udof] = mj_name2id(&_model, mjOBJ_JOINT, udof.c_str());

        for(unsigned int i = 0; i < _va.size(); ++i)
            _va_ids.push_back(_map_mujoco_joints.at(_va[i]));
        for(unsigned int i = 0; i < _vu.size(); ++i)
            _vu_ids.push_back(_map_mujoco_joints.at(_vu[i]));


        _U.setZero(_cols, _vu_ids.size());
        for(unsigned int i = 0; i < _vu_ids.size(); ++i)
            _U(_vu_ids[i], i) = 1.;
        _A.setZero(_cols, _va_ids.size());
        for(unsigned int i = 0; i < _va_ids.size(); ++i)
            _A(_va_ids[i], i) = 1.;


        for (int i = 0; i < _model.neq; ++i)
            _ceq.push_back(mj_id2name(&_model, mjOBJ_EQUALITY, i));


        _Ja.setZero(_nnz_rows.size(), _va.size());
        _Ju.setZero(_nnz_rows.size(), _vu.size());
        _Jm.setZero(_nnz_rows.size(), _va.size());

        _P.setZero(_va.size(), _vu.size());

        update();
    }

    /**
     * @brief getU return a matrix which maps passive quantities into generalized coordinates:
     *      qdot = U * qudot
     * @return mapping matrix
     */
    const MatrixX& getU() const { return _U;}

    /**
     * @brief getA return a matrix which maps active quantities into generalized coordinates
     *      qdot = A * qadot
     * @return mapping matrix
     */
    const MatrixX& getA() const { return _A;}

    /**
     * @brief setP compute the matrix which selects passive joints of interest such that:
     *      qjdot = P * qudot
     * @param joints list of passive joint names
     */
    void setP(const std::vector<std::string>& joints)
    {
        for(const auto& j : joints)
        {
            if(std::find(_va.begin(), _va.end(), j) != _va.end())
                throw std::runtime_error("joints should be passive not active in setP");
        }

        if(joints.size() != _P.rows())
            throw std::runtime_error("joints.size() != _P.rows()");

        for(unsigned int i = 0; i < joints.size(); ++i)
        {
            if(std::find(_vu.begin(), _vu.end(), joints[i]) == _vu.end())
                throw std::runtime_error(("Can not find joint "+joints[i]+" in passive dofs").c_str());

            ptrdiff_t j = std::distance(_vu.begin(), std::find(_vu.begin(), _vu.end(), joints[i]));

            _P(i, j) = 1.;
            _p.push_back(joints[i]);
        }
    }


    /**
     * @brief ~ClosedLinkage destructor
     */
    virtual ~ClosedLinkage(){}

    /**
     * @brief getID return closed linkage id
     * @return string
     */
    const std::string& getID() const { return _id; }

    /**
     * @brief getActiveDofs return a vector of string containing active dof names
     * @return vector
     */
    const std::vector<std::string>& getActiveDofs() const { return _va; }

    /**
     * @brief getPassiveDofs return a vector of string containing passive dof names
     * @return vector
     */
    const std::vector<std::string>& getPassiveDofs() const { return _vu; }

    /**
     * @brief toEigen maps an array into an eigen matrix
     * @param A matrix to be filled
     * @param data array of data
     * @param rows of the matrix
     * @param cols of the matrix
     */
    static void toEigen(MatrixX& A, T* data, unsigned int rows, unsigned int cols)
    {
        A = Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(data, rows, cols);
    }

    static void toEigen(VectorX& v, T* data, unsigned int rows)
    {
        v = Eigen::Map<Eigen::Matrix<T,Eigen::Dynamic,1> >(data, rows, 1);
    }

    /**
     * @brief print some info related to the closed linkage
     */
    void print()
    {
        mj_cll::print_closed_linkage_info(_id, _map_mujoco_joints,
                                          _va, _vu,
                                          _ceq, _nnz_rows,
                                          _efc_J_rank);
    }

    /**
     * @brief getRows
     * @return number of constraints rows
     */
    const unsigned int getRows() const {return _rows;}

    /**
     * @brief getNNZRows
     * @return number of non-zero rows
     */
    const unsigned int getNNZRows() const {return _nnz_rows.size();}

    /**
     * @brief getCols
     * @return columns of constraint matrix
     */
    const unsigned int getCols() const {return _cols;}

    /**
     * @brief getNNZRowsIndices()
     * @return vector of indices of non-zero rows in constraint matrix
     */
    const std::vector<unsigned int>& getNNZRowsIndices() const {return _nnz_rows;}

    /**
     * @brief getConstraintJacobian return the constraint Jacobian full rank
     * @return matrix
     */
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& getConstraintJacobian() const {return _J;}

    /**
     * @brief getPassivePartJacobian return the passive part of the constraint Jacobian
     * @return matrix
     * @note remember to call update()
     */
    const MatrixX& getPassivePartJacobian() const {return _Ju;}

    /**
     * @brief getActivePartJacobian return the actuated part of the constraint Jacobian
     * @return Matrix
     * @note remember to call update()
     */
    const MatrixX& getActivePartJacobian() const {return _Ja;}

    /**
     * @brief getMappingJacobian return the maping Jacobian to map actuated quantities into passive ones:
     *      qudot = Jm * qadot
     * @return matrix
     * @note rememeber to call update(true)
     */
    const MatrixX& getMappingJacobian() const {return _Jm;}

    /**
     * @brief update constraints Jacobian matrices and constraint error vector
     * @param compute_mapping_jacobian if true the mapping Jacobina is computed
     */
    void update(bool compute_mapping_jacobian = true)
    {
        _J.setZero(_nnz_rows.size(), _J.cols());
        copy_rows(_data.efc_J, _rows, _cols, _nnz_rows, _J);


        copy_columns(_J, _Ja, _va_ids);
        copy_columns(_J, _Ju, _vu_ids);

        unsigned int nnz_rows_size = _nnz_rows.size();
        for(unsigned int i = 0; i < nnz_rows_size; ++i)
            _ce[i] = _data.efc_pos[_nnz_rows[i]];

        if(compute_mapping_jacobian)
            computeMappingJacobian();
    }

    /**
     * @brief getConstraintError return vector of constraint error of the size of the constraint Jacobian
     * @return vector
     * @note remember to call update()
     */
    const VectorX& getConstraintError() const { return _ce; }

    /**
     * @brief getP return the matrix which selects passive joints of interest such that:
     *      qjdot = P * qdot
     * @return matrix
     */
    const MatrixX& getP() const { return _P; }

    /**
     * @brief getJointsP return vector of names used to compute P
     * @return vector of passive dof names
     */
    const std::vector<std::string>& getJointsP() const { return _p; }

    typedef std::string Joint;
    typedef int ModelID;
    /**
     * @brief getMujocoJointMap
     * @return map joint_names : mujoco_id
     */
    const std::map<Joint, ModelID>& getMujocoJointMap() const { return _map_mujoco_joints; }

    /**
     * @brief computeMappingJacobian the maping Jacobian to map actuated quantities into passive ones:
     *      qudot = Jm * qadot
     */
    void computeMappingJacobian()
    {
      if (_Ja.cols() == 0)
        throw std::runtime_error("_Ja.cols() == 0");
      if (_Ju.cols() == 0)
        throw std::runtime_error("_Ju.cols() == 0");

      _Jm = _Ju.householderQr().solve(-_Ja);
    }

private:
    std::vector<std::string> _va; // vector of active dofs
    std::vector<unsigned int> _va_ids;
    std::vector<std::string> _vu; // vector of passive dofs
    std::vector<unsigned int> _vu_ids;

    MatrixX _P; // mapping between joints of interest and passive dofs
    std::vector<std::string> _p; // passive dofs of interest

    M& _model;
    D& _data;

    std::map<Joint, ModelID> _map_mujoco_joints; // map joint_names : mujoco_id

    unsigned int _rows; // total number of rows of the constraint equality Jacobian
    std::vector<unsigned int> _nnz_rows; // indices of non-zero rows in constraint equality Jacobian
    unsigned int _cols; //number of cols of the constaint equality Jacobian

    std::vector<std::string> _ceq; // vector of equality constraints

    std::string _id;
    /**
     * @note Eigen defulat is column-major while mujoco default is row major!
     */
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _efc_J; // this is the full constraint jacobian copied from Mujoco
    unsigned int _efc_J_rank;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _J;         //  constraint Jacobian [l x n], ins this, zero rows are removed
    MatrixX _Ja;        // actuated part [l x na]
    MatrixX _Ju;        //  underactuated part [l x l = nu] square!
    MatrixX _Jm;        //  mapping Jacobian

    MatrixX _U; // maps underactuated quantities into full generalized coordiantes: dq = U*dqu, U [n x u]
    MatrixX _A; // maps actuated quantities into full generalized coordiantes: dq = A*dqa, A [n x a]

    VectorX _ce; // constraint error

    void copy_rows(const T* data, const unsigned int rows, const unsigned int cols, const std::vector<unsigned int>& rows_to_copy,
                   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A)
    {
        unsigned int rows_to_copy_size = rows_to_copy.size();
        for(unsigned int i = 0; i < rows_to_copy_size; ++i)
            copy_row(data, rows, cols, rows_to_copy[i], A, i);
    }



    /**
     * @brief copy_columns copy columns from A to B
     * @param A matrix row major
     * @param B matrix
     * @param cols vector of columns to copy
     */
    void copy_columns(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A,
                      MatrixX& B, const std::vector<unsigned int>& cols)
    {
        if(cols.size() > B.cols())
            throw std::runtime_error("cols.size() > B.cols()");
        if(A.rows() > B.rows())
            throw std::runtime_error("A.rows() > B.rows()");
        unsigned int cols_size = cols.size();
        for(unsigned int i = 0; i < cols_size; ++i)
            B.col(i) = A.col(cols[i]);
    }

    /**
     * @brief copy_row Copies a row from mujoco  matrix type to MatrixX
     * @param data mujoco matrix
     * @param rows size of mujoco matrix
     * @param cols size of mujoco matrix
     * @param row_to_copy row id of mujoco matrix to copy in the MatrixX
     * @param copy_to_row id of the row of MatrixX where to copy mujoco row
     * @param A row-major eigen matrix
     */
    void copy_row(const T* data, const unsigned int rows, const unsigned int cols, const unsigned int row_to_copy,
                  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A, const unsigned int copy_to_row)
    {
        if(copy_to_row > A.rows())
            throw std::runtime_error("copy_to_row > A.rows()");
        if(row_to_copy > rows)
            throw std::runtime_error("row_to_copy > rows");
        if(cols > A.cols())
            throw std::runtime_error("cols > A.cols()");

        std::memcpy(A.row(copy_to_row).data(), data + cols*row_to_copy, cols * sizeof(data[0]));
    }


    /**
     * @brief compute_constraints_rows_indices given a matrix in input, it checks for zero rows and equal rows removing associated indices so that the output is a list of rows indices
     * that can be used to create a new matrix from the initial one where there are no zero rows neither same rows.
     * A check is performed against the rank of the original matrix because we expect that the number of constraints is n-a = u, with:
     * - n: total number of degrees of freedom (dofs)
     * - a: active dofs
     * - u: passive dofs
     * @note the rank depends also on linear dependent rows which however is not tested. We should perform some linear dependency test such as QR decomposition
     * @todo try more rigorous way to compute rows of constraints according to rank of the original constraint matrix
     * @param cJfull full constraint jacobian from Mujoco
     * @param constraint_rows output rows (zeros and same rows removed)
     * @param rank_cJfull rank of full constraint jacobian
     * @param row_zero_ths threshold to check zero rows (very sensitive!)
     * @param rank_ths threshold to compute rank (very sensitive!)
     * @param is_similar_ths threshold to check if two rows are similar (very sensitive!)
     */
    void compute_constraints_rows_indices(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& cJfull,
                                          std::vector<unsigned int>& constraint_rows, unsigned int& rank_cJfull,
                                          const double row_zero_ths, const double rank_ths, const double is_similar_ths)
    {
        //1. we check the rank of cJfull
        Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lu_decomp(cJfull);
        lu_decomp.setThreshold(rank_ths);

        rank_cJfull = lu_decomp.rank();

        //2. we remove rows that are zero
        for(unsigned int r = 0; r < cJfull.rows(); ++r)
        {
            if(!cJfull.row(r).isZero(row_zero_ths))
                constraint_rows.push_back(r);
        }

        //3. we check rows that are the same and remove them as well
        std::map<unsigned int, std::vector<unsigned int> > same_rows_map;
        for(auto ri : constraint_rows)
        {
            for(auto rj : constraint_rows)
            {
                std::vector<unsigned int> same_rows;
                if(ri != rj)
                {
                    if(cJfull.row(ri).isApprox(cJfull.row(rj), is_similar_ths))
                    {
                        if(!same_rows_map.count(rj))
                            same_rows.push_back(rj);
                    }
                }
                if(same_rows.size() > 0)
                    same_rows_map[ri] = same_rows;

            }
        }

        for ( const auto &p : same_rows_map )
        {
            for(auto id : p.second)
                constraint_rows.erase(std::remove(constraint_rows.begin(), constraint_rows.end(), id), constraint_rows.end());
        }

        if(constraint_rows.size() != rank_cJfull)
        {
            std::stringstream ss;
            ss<< "constraint_rows.size() != rank_cJfull : constraint_rows.size() is "<<constraint_rows.size()<<" while rank_cJfull is "<<rank_cJfull<<"."<<std::endl;
            ss<<" Consider to play with row_zero_ths, rank_ths, and is_similar_ths."<<std::endl;
            throw std::runtime_error(ss.str().c_str());
        }
    }


};

typedef ClosedLinkage<mjtNum, mjModel, mjData> ClosedLinkageMj;

}

#endif
