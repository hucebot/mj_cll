#ifndef _MJ_CLL_UTILS_H_
#define _MJ_CLL_UTILS_H_

#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include <map>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>



namespace mj_cll
{

    /**
     * @brief get_joint_names returns ordered joint names from mujoco model
     * @param model
     * @param joint_names
     */
    void get_joint_names(const mjModel* model, std::vector<std::string>& joint_names);


    /**
     * @brief print_closed_linkage_info prints basic informations from the closed linkage to terminal
     * @param id of the closed linkage
     * @param map_mujoco_joints print the map-> joint_name : mujoco_id
     * @param active_dofs vector of names of active dofs
     * @param passive_dofs vector of names of passive dofs
     * @param equality_constraints vector of names of equality constraints
     * @param non_zero_rows vector of id of non-zero rows
     * @param efc_J_rank rank of original constraint Jacobian
     */
    void print_closed_linkage_info(const std::string& id, const std::map<std::string, int>& map_mujoco_joints,
                                   const std::vector<std::string>& active_dofs, const std::vector<std::string>& passive_dofs,
                                   const std::vector<std::string>& equality_constraints, const std::vector<unsigned int>& non_zero_rows,
                                   const int efc_J_rank);


}

#endif
