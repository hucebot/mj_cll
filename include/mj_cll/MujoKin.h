#ifndef _MJ_CLL_MUJOKIN_H_
#define _MJ_CLL_MUJOKIN_H_

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <functional>
#include <unistd.h>
#include <exception>
#include <stdexcept>
#include <chrono>

namespace mj_cll{

class MujoKin
{
public:
    MujoKin();
    void createScene(const mjModel *model);
    void show(const mjModel* model, mjData *data);
    void loop(const mjModel* model, mjData *data, std::function<void()> func,
              const useconds_t us = 0., const std::chrono::milliseconds stop_in = std::chrono::milliseconds(-1));

    GLFWwindow* getWindow(){ return _window; }

    ~MujoKin();

private:

    mjvCamera _cam;                      // abstract camera
    mjvOption _opt;                      // visualization options
    mjvScene _scn;                       // abstract scene
    mjrContext _con;                     // custom GPU context
    mjvPerturb _pert;
    GLFWwindow* _window;

    std::exception_ptr _eptr;
    void handle_eptr(std::exception_ptr eptr);

    static void mouse_button(GLFWwindow* window, int button, int act, int mods);
    static void scroll(GLFWwindow* window, double xoffset, double yoffset);
    static void mouse_move(GLFWwindow* window, double xpos, double ypos);

    static bool _button_left;
    static bool _button_middle;
    static bool _button_right;
    static double _lastx;
    static double _lasty;
    static double _zoom;
    static int _width, _height;
    static double _dx, _dy;
    static mjtMouse _action;

};

}

#endif
