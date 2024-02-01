#include <mj_cll/MujoKin.h>
#include <iostream>

using namespace mj_cll;

bool MujoKin::_button_left = false;
bool MujoKin::_button_middle = false;
bool MujoKin::_button_right = false;
double MujoKin::_lastx = 0.;
double MujoKin::_lasty = 0.;
double MujoKin::_zoom = 0.;
double MujoKin::_dx = 0.;
double MujoKin::_dy = 0.;
mjtMouse MujoKin::_action = mjMOUSE_ZOOM;
int MujoKin::_height = 0;
int MujoKin::_width = 0;

MujoKin::MujoKin()
{
    // init GLFW, create window, make OpenGL context current, request v-sync
    glfwInit();
    _window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);

    glfwSetCursorPosCallback(_window, MujoKin::mouse_move);
    glfwSetMouseButtonCallback(_window, MujoKin::mouse_button);
    glfwSetScrollCallback(_window, MujoKin::scroll);

    // initialize visualization data structures
    mjv_defaultCamera(&_cam);
    mjv_defaultPerturb(&_pert);
    mjv_defaultOption(&_opt);
    mjr_defaultContext(&_con);
    mjv_defaultScene(&_scn);
}

void MujoKin::createScene(const mjModel *model)
{
    // create scene and context
    mjv_makeScene(model, &_scn, 1000);
    mjr_makeContext(model, &_con, mjFONTSCALE_100);
}

void MujoKin::show(const mjModel *model, mjData* data)
{
    // get current window size
    glfwGetWindowSize(_window, &MujoKin::_width, &MujoKin::_height);

    // get framebuffer viewport
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(_window, &viewport.width, &viewport.height);

      mjv_moveCamera(model, mjMOUSE_ZOOM, 0, _zoom, &_scn, &_cam);
      MujoKin::_zoom = 0.;
      mjv_moveCamera(model, MujoKin::_action, MujoKin::_dx/MujoKin::_height, MujoKin::_dy/MujoKin::_height, &_scn, &_cam);
      _dx = _dy = 0.;

      // update scene and render
      mjv_updateScene(model, data, &_opt, NULL, &_cam, mjCAT_ALL, &_scn);
      mjr_render(viewport, &_scn, &_con);

      // swap OpenGL buffers (blocking call due to v-sync)
      glfwSwapBuffers(_window);

      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();
}

MujoKin::~MujoKin()
{
    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&_scn);
    mjr_freeContext(&_con);
}

void MujoKin::handle_eptr(std::exception_ptr eptr) // passing by value is ok
{
    try
    {
        if (eptr)
            std::rethrow_exception(eptr);
    }
    catch(const std::exception& e)
    {
        std::cout << "Caught exception: '" << e.what() << "'\n";
    }
}

void MujoKin::loop(const mjModel* model, mjData *data, std::function<void()> func, const useconds_t us,
                   const std::chrono::milliseconds stop_in)
{
    bool timer_start = false;
    std::chrono::time_point<std::chrono::system_clock> beg, end;
    if(stop_in > std::chrono::milliseconds(0))
    {
        timer_start = true;
        beg = std::chrono::high_resolution_clock::now();
    }

    while(!glfwWindowShouldClose(_window))
    {
        if(func != NULL)
        {
            try
            {
                func();
            }
            catch(...)
            {
                _eptr = std::current_exception(); // capture
            }
        }

        handle_eptr(_eptr);
        show(model, data);
        usleep(us);

        if(timer_start)
        {
            end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
            if(duration >= stop_in)
                break;
        }
    }
}

void MujoKin::mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  MujoKin::_button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  MujoKin::_button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  MujoKin::_button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &MujoKin::_lastx, &MujoKin::_lasty);
}

void MujoKin::scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  MujoKin::_zoom = -0.05*yoffset;
}

void MujoKin::mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!MujoKin::_button_left && !MujoKin::_button_middle && !MujoKin::_button_right) {
    return;
  }

  // compute mouse displacement, save
  MujoKin::_dx = xpos - MujoKin::_lastx;
  MujoKin::_dy = ypos - MujoKin::_lasty;
  MujoKin::_lastx = xpos;
  MujoKin::_lasty = ypos;

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  if (MujoKin::_button_right) {
    MujoKin::_action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (MujoKin::_button_left) {
    MujoKin::_action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    MujoKin::_action = mjMOUSE_ZOOM;
  }
}
