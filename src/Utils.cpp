#include <mj_cll/Utils.h>
#include <iostream>


void mj_cll::get_joint_names(const mjModel* model, std::vector<std::string>& joint_names)
{
    for (int i = 0; i < model->njnt; ++i)
        joint_names.push_back(mj_id2name(model, mjOBJ_JOINT, i));
}

void mj_cll::print_closed_linkage_info(const std::string& id, const std::map<std::string, int>& map_mujoco_joints,
                               const std::vector<std::string>& active_dofs, const std::vector<std::string>& passive_dofs,
                               const std::vector<std::string>& equality_constraints, const std::vector<unsigned int>& non_zero_rows,
                               const int efc_J_rank)
{
    std::cout<<"ID: "<<id<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Mujoco joint map: "<<std::endl;
    for(auto& mj : map_mujoco_joints)
        std::cout<<"    "<<mj.first<<" : "<<mj.second<<std::endl;
    std::cout<<std::endl;

    std::cout<<"active_dofs: [ ";
    for(auto& a : active_dofs)
        std::cout<<a<<" ";
    std::cout<<"]"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"passive_dofs: [ ";
    for(auto& u : passive_dofs)
        std::cout<<u<<" ";
    std::cout<<"]"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Equality constraints: "<<std::endl;
    for(auto& eq : equality_constraints)
        std::cout<<"    "<<eq<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Non-zero rows indices: "<<std::endl;
    for(auto& id : non_zero_rows)
        std::cout<<"    "<<id<<std::endl;
    std::cout<<std::endl;

    std::cout<<"_efc_J_rank: "<<efc_J_rank<<std::endl;
    std::cout<<std::endl;
}


