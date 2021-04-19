import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from numba import njit 
import time

G = 9.81

class Thruster():

    def __init__(self, max_force=10):
        self._force = 0
        self._angle = 0
        self._throtle = 1
        self._max_force = max_force

    @property
    def throtle(self):
        return self._throtle

    @throtle.setter
    def throtle(self, value):
        if value < -1:
            value = -1
        elif value > 1:
            value = 1
        self._throtle = value
        self._force = self._throtle * self._max_force

    @property
    def force(self):
        return np.array([self._force*np.sin(self._angle), self._force*np.cos(self._angle)])

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
         
    

class Copter():

    def __init__(self, pos=[0, 0], orientation=0):
        self.mass = 1
        self.arm_right = 0.5
        self.arm_left = 0.5
        self.orientation = np.deg2rad(orientation)
        self.pos = np.array(pos, dtype="float64")
        self.thruster_right = Thruster(max_force=40)
        self.thruster_left = Thruster(max_force=40)
        self.thruster_right.angle = np.deg2rad(self.orientation)
        self.thruster_left.angle = np.deg2rad(self.orientation)
        self.thruster_right.throtle = 0
        self.thruster_left.throtle = 0
        self.moment_of_inertia = 10

        self.velocity = np.array([0., 0.])
        self.rotational_acceleration = 0
        self.rotational_velocity = 0


        self.pos_x_plot = []
        self.pos_y_plot = []
        self.orientation_plot = []
        self.thruster_right_throtle_plot = []
        self.thruster_left_throtle_plot = []
        self.thruster_right_orientation_plot = []
        self.thruster_left_orientation_plot = []
        self.time_plot = []

    def __totat_forces(self):
        gravity_force = np.array([0, -self.mass*G])
        force_body = gravity_force + self.thruster_left.force + self.thruster_right.force
        force = self.__rotate_vector(force_body, np.deg2rad(self.orientation))
        # print(f"force = {force}")
        return force

    def __rotate_vector(self, vec, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
        return np.matmul(rot_mat, vec)

    def __total_moments(self):
        moments_body = self.thruster_right.force[1] * self.arm_right - self.thruster_left.force[1] * self.arm_left
        return moments_body 

    def update(self, dt = 0.0001):
        self.acceleration = self.__totat_forces() / self.mass
        self.velocity += self.acceleration * dt
        self.pos += self.velocity * dt
        # self.thruster_right.throtle = -(self.pos[1]) / -self.velocity[1]
        # self.thruster_left.throtle = -(self.pos[1]) / -self.velocity[1]*2

        self.rotational_acceleration = self.__total_moments() / self.moment_of_inertia
        self.rotational_velocity += self.rotational_acceleration * dt
        self.orientation += self.rotational_velocity * dt


    def control(self, throtle_right = 0, throtle_left = 0, angle_right = 0, angle_left = 0):
        self.thruster_right.throtle = throtle_right
        self.thruster_right.angle = angle_right
        self.thruster_left.throtle = throtle_left
        self.thruster_left.angle = angle_left

    def getState(self):
        return {"pos": self.pos, 
                "orientation": self.orientation, 
                "velocity": self.velocity, 
                "rotational_velocity": self.rotational_velocity}

    def plot(self, fig, t):

        fig.clf()
        fig.suptitle(f'Time = {round(t,2)} s', fontsize=16)
        self.pos_x_plot.append(self.pos[0])
        self.pos_y_plot.append(self.pos[1])
        self.orientation_plot.append(self.orientation)
        self.thruster_right_throtle_plot.append(self.thruster_right.throtle)
        self.thruster_left_throtle_plot.append(self.thruster_left.throtle)
        self.thruster_right_orientation_plot.append(self.thruster_right.angle)
        self.thruster_left_orientation_plot.append(self.thruster_left.angle)
        self.time_plot.append(t)
        # print(self.pos_plot[1])
        gs = GridSpec(nrows=2, ncols=2)
        ax0 = fig.add_subplot(gs[0, 1])
        ax0.plot(self.time_plot, self.orientation_plot)
        ax1 = fig.add_subplot(gs[1, 1])
        ax1.plot(self.time_plot, self.thruster_right_throtle_plot)
        ax1.plot(self.time_plot, self.thruster_left_throtle_plot)
        ax2 = fig.add_subplot(gs[:, 0])

        
        ax2.axis("equal")
        ax2.grid()
        ax2.plot([-5, 5, -5, 5], [-5, -5, 5, 5], "r.")
        ax2.plot([self.pos_x_plot[-1]+np.cos(self.orientation)*self.arm_right, self.pos_x_plot[-1]-np.cos(self.orientation)*self.arm_left], 
                 [self.pos_y_plot[-1]+np.sin(self.orientation)*self.arm_right, self.pos_y_plot[-1]-np.sin(self.orientation)*self.arm_left], "b*-")


        # ax2.plot(self.pos_x_plot, self.pos_y_plot, "b*")
        plt.draw()
        plt.pause(0.01)



class Controler():

    def __init__(self, drone):
        self.drone = drone
        self.err_prev = 0

        self.Kp = -10.1
        self.Kd = -1
        self.Ki = -10

    def controlIt(self, point, dt):
        
        targetPoint = np.array(point)


        if drone.pos[1] > targetPoint[1]:
            err = np.sqrt((targetPoint[0] - self.drone.pos[0])**2 + (targetPoint[1] - self.drone.pos[1])**2)
        else:
            err = -np.sqrt((targetPoint[0] - self.drone.pos[0])**2 + (targetPoint[1] - self.drone.pos[1])**2)
            
        err_derivative = (err - self.err_prev)/dt
        err_integral = dt*(err + self.err_prev)/2

        self.err_prev = err

        res = self.Kp*err + self.Kd*err_derivative + self.Ki*err_integral
        self.drone.control(throtle_right=res, throtle_left = res, angle_left=0, angle_right=0)





if __name__ == "__main__":
    
    drone = Copter(pos=[0, 0], orientation=0)
    controler = Controler(drone = drone)

    dt = 0.01
    t=0

    fig = plt.figure(figsize=(10, 5))
    while t < 1000:
        t0 = time.time()

        drone.update(dt = dt)
        if t < 10:
            controler.controlIt([0,0], dt)
        elif t<20:
            controler.controlIt([0, 5], dt)
        elif t<4:
            controler.controlIt([0, 0], dt)

        dt = 0.01
        dt = time.time() - t0

        drone.plot(fig, t)

        t += dt