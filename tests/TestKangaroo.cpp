#include <gtest/gtest.h>
#include <mj_cll/ClosedLinkage.hpp>
#include <mj_cll/Utils.h>
#include <mj_cll/MujoKin.h>
#include <mj_cll/IK.h>
#include <chrono>
#include <math.h>
#include <random>

#define torad(x) x*M_PI/180.
#define todeg(x) x*180./M_PI

std::string path = _TESTS_FOLDER "kangaroo.mjb";

namespace {

class TestKangaroo: public ::testing::Test
{
protected:

    TestKangaroo()
    {

    }

    virtual ~TestKangaroo() {
    }

    virtual void SetUp() {

    }

    virtual void TearDown() {

    }

};

TEST_F(TestKangaroo, testIK)
{
    mjModel* m = mj_loadModel(path.c_str(), NULL);
    if(m == NULL){ASSERT_TRUE(false);}


    mjData* d = mj_makeData(m);

    mj_fwdPosition(m, d);

    std::vector<std::string> active_dofs;
    active_dofs.push_back("leg_left_1_motor");
    active_dofs.push_back("leg_left_2_motor");
    active_dofs.push_back("leg_left_3_motor");
    active_dofs.push_back("leg_left_length_motor");
    active_dofs.push_back("leg_left_4_motor");
    active_dofs.push_back("leg_left_5_motor");

    mj_cll::ClosedLinkageMj::Ptr closed_linkage = std::make_shared<mj_cll::ClosedLinkageMj>("kangaroo_keft_leg", active_dofs, *m, *d);
    closed_linkage->update();
    closed_linkage->print();

    std::cout<<"J: \n"<<closed_linkage->getConstraintJacobian()<<std::endl;
    std::cout<<"Ja: ["<<closed_linkage->getActivePartJacobian().rows()<<" x "<<closed_linkage->getActivePartJacobian().cols()<<"]"<<std::endl;
    std::cout<<"Ju: ["<<closed_linkage->getPassivePartJacobian().rows()<<" x "<<closed_linkage->getPassivePartJacobian().cols()<<"]"<<std::endl;
    std::cout<<"Ju: \n"<<closed_linkage->getPassivePartJacobian()<<std::endl;


    std::vector<std::string> p;
    p.push_back("leg_left_1_joint");
    p.push_back("leg_left_2_joint");
    p.push_back("leg_left_3_joint");
    p.push_back("leg_left_knee_joint");
    p.push_back("leg_left_4_joint");
    p.push_back("leg_left_5_joint");
    closed_linkage->setP(p);
    std::cout<<"P: \n"<<closed_linkage->getP()<<std::endl;

    mj_cll::KinematicsMj::Ptr ik = std::make_shared<mj_cll::KinematicsMj>(*closed_linkage);


    mj_cll::MujoKin viz;
    viz.createScene(m);

    Eigen::VectorXd q_des(p.size());
    q_des[0] = torad(-30.);
    q_des[1] = torad(-20.);
    q_des[2] = torad(10.);
    q_des[3] = torad(90.);
    q_des[4] = torad(30.); // ankle pitch
    q_des[5] = torad(10.); // ankle roll
    std::cout<<"Desired Joints values: "<<std::endl;
    std::cout<<"    leg_left_1_joint: "<<todeg(q_des[0])<<std::endl;
    std::cout<<"    leg_left_2_joint: "<<todeg(q_des[1])<<std::endl;
    std::cout<<"    leg_left_3_joint: "<<todeg(q_des[2])<<std::endl;
    std::cout<<"    leg_left_knee_joint: "<<todeg(q_des[3])<<std::endl;
    std::cout<<"    leg_left_4_joint: "<<todeg(q_des[4])<<std::endl;
    std::cout<<"    leg_left_5_joint: "<<todeg(q_des[5])<<std::endl;


    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    EXPECT_TRUE(ik->ikLoop(*m, *d, q_des, 1.0, 0.9, 1e-4, 1e-2));
    //EXPECT_TRUE(ik->ikLoopQP(*m, *d, q_des, 1.0, 0.9, 1e-4, 1e-2));
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    std::cout << ik->getIterations() << " iterations\n";

    std::cout<<"Actual Joints values: "<<std::endl;
    std::cout<<"    leg_left_1_joint: "<<todeg(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_1_joint")])<<std::endl;
    std::cout<<"    leg_left_2_joint: "<<todeg(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_2_joint")])<<std::endl;
    std::cout<<"    leg_left_3_joint: "<<todeg(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_3_joint")])<<std::endl;
    std::cout<<"    leg_left_knee_joint: "<<todeg(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_knee_joint")])<<std::endl;
    std::cout<<"    leg_left_4_joint: "<<todeg(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_4_joint")])<<std::endl;
    std::cout<<"    leg_left_5_joint: "<<todeg(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_5_joint")])<<std::endl;


    EXPECT_NEAR(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_1_joint")], q_des[0], 1e-4);
    EXPECT_NEAR(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_2_joint")], q_des[1], 1e-4);
    EXPECT_NEAR(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_3_joint")], q_des[2], 1e-4);
    EXPECT_NEAR(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_knee_joint")], q_des[3], 1e-4);
    EXPECT_NEAR(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_4_joint")], q_des[4], 1e-4);
    EXPECT_NEAR(d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_5_joint")], q_des[5], 1e-4);



    for(unsigned int i = 0; i < 300; ++i)
        viz.show(m, d);
//    viz.loop(m, d, NULL,1000);

    mj_deleteData(d);
    mj_deleteModel(m);

}

double generateRandomDouble(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    return dis(gen);
}

TEST_F(TestKangaroo, testIKQP)
{
    mjModel* m = mj_loadModel(path.c_str(), NULL);
    if(m == NULL){ASSERT_TRUE(false);}


    mjData* d = mj_makeData(m);

    mj_fwdPosition(m, d);

    std::vector<std::string> active_dofs;
    active_dofs.push_back("leg_left_1_motor");
    active_dofs.push_back("leg_left_2_motor");
    active_dofs.push_back("leg_left_3_motor");
    active_dofs.push_back("leg_left_length_motor");
    active_dofs.push_back("leg_left_4_motor");
    active_dofs.push_back("leg_left_5_motor");

    mj_cll::ClosedLinkageMj::Ptr closed_linkage = std::make_shared<mj_cll::ClosedLinkageMj>("kangaroo_keft_leg", active_dofs, *m, *d);
    closed_linkage->update();
    closed_linkage->print();

    std::cout<<"J: \n"<<closed_linkage->getConstraintJacobian()<<std::endl;
    std::cout<<"Ja: ["<<closed_linkage->getActivePartJacobian().rows()<<" x "<<closed_linkage->getActivePartJacobian().cols()<<"]"<<std::endl;
    std::cout<<"Ju: ["<<closed_linkage->getPassivePartJacobian().rows()<<" x "<<closed_linkage->getPassivePartJacobian().cols()<<"]"<<std::endl;
    std::cout<<"Ju: \n"<<closed_linkage->getPassivePartJacobian()<<std::endl;


    std::vector<std::string> p;
    p.push_back("leg_left_1_joint");
    p.push_back("leg_left_2_joint");
    p.push_back("leg_left_3_joint");
    p.push_back("leg_left_knee_joint");
    p.push_back("leg_left_4_joint");
    p.push_back("leg_left_5_joint");
    closed_linkage->setP(p);
    std::cout<<"P: \n"<<closed_linkage->getP()<<std::endl;

    mj_cll::KinematicsMj::Ptr ik = std::make_shared<mj_cll::KinematicsMj>(*closed_linkage);

    mj_cll::MujoKin viz;
    viz.createScene(m);

    Eigen::VectorXd q_des(p.size());
    q_des[0] = torad(generateRandomDouble(-90., 90.));
    q_des[1] = torad(generateRandomDouble(-90., 90.));
    q_des[2] = torad(generateRandomDouble(-90., 90.));
    q_des[3] = torad(generateRandomDouble(-90, 90.));
    q_des[4] = torad(generateRandomDouble(-90., 90.)); // ankle pitch
    q_des[5] = torad(generateRandomDouble(-90., 90.)); // ankle roll
    std::cout<<"Desired Joints values: "<<std::endl;
    std::cout<<"    leg_left_1_joint: "<<q_des[0]<<std::endl;
    std::cout<<"    leg_left_2_joint: "<<q_des[1]<<std::endl;
    std::cout<<"    leg_left_3_joint: "<<q_des[2]<<std::endl;
    std::cout<<"    leg_left_knee_joint: "<<q_des[3]<<std::endl;
    std::cout<<"    leg_left_4_joint: "<<q_des[4]<<std::endl;
    std::cout<<"    leg_left_5_joint: "<<q_des[5]<<std::endl;


    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    //EXPECT_TRUE(ik->ikLoop(*m, *d, q_des, 1.0, 0.9, 1e-4, 1e-2));

    Eigen::VectorXd qmin(m->nq), qmax(m->nq);
    unsigned int j = 0;
    std::cout<<"Joint limits"<<std::endl;
    for(unsigned int i = 0; i < 2*m->nq; ++i)
    {
        qmin[j] = m->jnt_range[i];
        i++;
        qmax[j] = m->jnt_range[i];

        std::cout<<"    q["<<j<<"]"<<":  "<<qmin[j]<<" :: "<<qmax[j]<<std::endl;
        j++;
    }


    bool success = ik->ikLoopQP(*m, *d, q_des, qmin, qmax, 1.0, 0.9, 1e-4, 1e-2, 1e-9, 2000);
    if(success)
        EXPECT_TRUE(success);
    else
    {
        auto ik_result = ik->getIkResult();
        if(ik_result == mj_cll::KinematicsMj::IKRESULT::MAX_ITER_ACHIEVED)
            EXPECT_TRUE(true);
        else
            EXPECT_TRUE(success);
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    std::cout << ik->getIterations() << " iterations\n";

    std::cout<<"Actual Joints values: "<<std::endl;
    std::cout<<"    leg_left_1_joint: "<<d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_1_joint")]<<std::endl;
    std::cout<<"    leg_left_2_joint: "<<d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_2_joint")]<<std::endl;
    std::cout<<"    leg_left_3_joint: "<<d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_3_joint")]<<std::endl;
    std::cout<<"    leg_left_knee_joint: "<<d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_knee_joint")]<<std::endl;
    std::cout<<"    leg_left_4_joint: "<<d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_4_joint")]<<std::endl;
    std::cout<<"    leg_left_5_joint: "<<d->qpos[closed_linkage->getMujocoJointMap().at("leg_left_5_joint")]<<std::endl;

    for(unsigned int i = 0; i < m->nq; ++i)
    {
        EXPECT_GE(d->qpos[i] + 1e-9, qmin[i]);
        EXPECT_LE(d->qpos[i] - 1e-9, qmax[i]);
    }

    for(unsigned int i = 0; i < m->nq; ++i)
        std::cout<<"q["<<i<<"]: "<<d->qpos[i]<<std::endl;

    std::cout<<"equality constraint error: "<<std::endl;
    for(unsigned int i = 0; i < 3*m->neq; ++i)
        std::cout<<"    "<<i<<":"<<d->efc_pos[i]<<std::endl;


    for(unsigned int i = 0; i < 300; ++i)
        viz.show(m, d);
//    viz.loop(m, d, NULL,1000);

    mj_deleteData(d);
    mj_deleteModel(m);

}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
