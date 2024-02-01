#include <gtest/gtest.h>
#include <mj_cll/ClosedLinkage.hpp>
#include <mj_cll/Utils.h>
#include <mj_cll/MujoKin.h>
#include <mj_cll/IK.h>
#include <chrono>

std::string path = _TESTS_FOLDER "pantograph.xml";


namespace {

class TestClosedLinkage: public ::testing::Test
{
protected:

    TestClosedLinkage()
    {

    }

    virtual ~TestClosedLinkage() {
    }

    virtual void SetUp() {

    }

    virtual void TearDown() {

    }

};

class random_q
{
public:
    random_q(mjModel* m, mjData*d):
        _m(m),
        _d(d)
    {

    }

    void random()
    {
        _d->qpos[0] = std::rand();
        _d->qpos[1] = std::rand();
        _d->qpos[2] = std::rand();
        mj_fwdPosition(_m, _d);
        Eigen::VectorXd e = Eigen::Map<Eigen::Vector3d, Eigen::Unaligned>(_d->efc_pos, 3);
        std::cout<<"equality constraint error: "<<e.transpose()<<std::endl;

    }

private:
    mjModel* _m;
    mjData* _d;
};


TEST_F(TestClosedLinkage, testLoadModel)
{
    char error[1000];
    mjModel* m = mj_loadXML(path.c_str(), NULL, error, 1000);

    if(m == NULL){
        std::cout<<std::string(error)<<std::endl;
        ASSERT_TRUE(false);}


    mjData* d = mj_makeData(m);

    mj_fwdPosition(m, d);

    mj_cll::MujoKin viz;
    viz.createScene(m);

    Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(d->qpos, m->nq);
    std::cout<<"q: "<<q.transpose()<<std::endl;

    for(unsigned int i = 0; i < 100; ++i)
        viz.show(m, d);

    Eigen::Vector3d e(d->efc_pos);
    std::cout<<"equality constraint error: "<<e.transpose()<<std::endl;

    random_q randq(m, d);

    viz.loop(m, d, std::bind(&random_q::random, randq), 100000, std::chrono::milliseconds(1000));

    if(d == NULL){
        std::cout<<"Problem making data from model"<<std::endl;
        ASSERT_TRUE(false);}

    // active dofs names
    std::vector<std::string> active_dofs;
    active_dofs.push_back("joint2");
    //active_dofs.push_back("joint5");



    mj_cll::ClosedLinkageMj::Ptr closed_linkage = std::make_shared<mj_cll::ClosedLinkageMj>("pantograph", active_dofs, *m, *d);
    closed_linkage->print();

    Eigen::MatrixXd Jce(closed_linkage->getRows(), closed_linkage->getCols());
    Jce.setZero();
    mj_cll::ClosedLinkageMj::toEigen(Jce, d->efc_J, closed_linkage->getRows(), closed_linkage->getCols());
    std::cout<<"Jce: "<<std::endl;
    std::cout<<Jce<<std::endl;


    std::cout<<"Jl: "<<std::endl;
    std::cout<<closed_linkage->getConstraintJacobian()<<std::endl;

    std::cout<<"Jlu: "<<std::endl;
    std::cout<<closed_linkage->getPassivePartJacobian()<<std::endl;

    std::cout<<"Jla: "<<std::endl;
    std::cout<<closed_linkage->getActivePartJacobian()<<std::endl;

    std::cout<<"Jm: "<<std::endl;
    std::cout<<closed_linkage->getMappingJacobian()<<std::endl;

    mj_deleteData(d);
    mj_deleteModel(m);

}

TEST_F(TestClosedLinkage, testIKLoop)
{
    char error[1000];
    mjModel* m = mj_loadXML(path.c_str(), NULL, error, 1000);

    if(m == NULL){ std::cout<<std::string(error)<<std::endl; ASSERT_TRUE(false);}

    mjData* d = mj_makeData(m);

    mj_fwdPosition(m, d);

    std::vector<std::string> active_dofs;
    active_dofs.push_back("joint2");

    mj_cll::ClosedLinkageMj::Ptr closed_linkage = std::make_shared<mj_cll::ClosedLinkageMj>("pantograph", active_dofs, *m, *d);

    std::vector<std::string> p; p.push_back("joint3");
    closed_linkage->setP(p);
    std::cout<<"P: "<<std::endl;
    std::cout<<closed_linkage->getP()<<std::endl;


    mj_cll::KinematicsMj::Ptr ik = std::make_shared<mj_cll::KinematicsMj>(*closed_linkage);

    Eigen::VectorXd q_des(p.size());
    q_des.setOnes();
    Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(d->qpos, m->nq);


    mj_cll::MujoKin viz;
    viz.createScene(m);

    double e = q_des[0] - q[closed_linkage->getMujocoJointMap().at("joint3")];
    while(e > 1e-6)
    {
        e = q_des[0] - q[closed_linkage->getMujocoJointMap().at("joint3")];
        closed_linkage->update();

        std::cout<<"e: "<<e<<std::endl;

        Eigen::VectorXd Ke(p.size());
        Ke[0] = 0.1*e;

        Eigen::VectorXd dqa;
        ik->computeIKVelocity(Ke, dqa);

        Eigen::VectorXd dqu = closed_linkage->getMappingJacobian()*dqa;

        Eigen::VectorXd dq(q.size());
        for(unsigned int i = 0; i < closed_linkage->getActiveDofs().size(); ++i)
            dq[closed_linkage->getMujocoJointMap().at(closed_linkage->getActiveDofs()[i])] = dqa[i];
        for(unsigned int i = 0; i < closed_linkage->getPassiveDofs().size(); ++i)
            dq[closed_linkage->getMujocoJointMap().at(closed_linkage->getPassiveDofs()[i])] = dqu[i];

        q += dq;

        for(unsigned int i = 0; i < q.size(); ++i)
            d->qpos[i] = q[i];

        mj_fwdPosition(m, d);

        viz.show(m, d);
        usleep(10000);
    }

    mj_deleteData(d);
    mj_deleteModel(m);
}

TEST_F(TestClosedLinkage, testIK)
{
    char error[1000];
    mjModel* m = mj_loadXML(path.c_str(), NULL, error, 1000);

    if(m == NULL){ std::cout<<std::string(error)<<std::endl; ASSERT_TRUE(false);}

    mjData* d = mj_makeData(m);

    mj_fwdPosition(m, d);

    std::vector<std::string> active_dofs;
    active_dofs.push_back("joint2");

    mj_cll::ClosedLinkageMj::Ptr closed_linkage = std::make_shared<mj_cll::ClosedLinkageMj>("pantograph", active_dofs, *m, *d);

    std::vector<std::string> p; p.push_back("joint3");
    closed_linkage->setP(p);

    mj_cll::KinematicsMj::Ptr ik = std::make_shared<mj_cll::KinematicsMj>(*closed_linkage);

    mj_cll::MujoKin viz;
    viz.createScene(m);

    Eigen::VectorXd q_des(p.size());
    q_des.setOnes();

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    EXPECT_TRUE(ik->ikLoop(*m, *d, q_des, 1.0, 0.01, 1e-4, 1e-4));
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    for(unsigned int i = 0; i < 300; ++i)
        viz.show(m, d);

    mj_deleteData(d);
    mj_deleteModel(m);
}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
