"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

from ntpath import join
from re import U
from matplotlib.pyplot import thetagrids
import numpy as np
# from torch import alpha_dropout
# expm is a matrix exponential function
from scipy.linalg import expm
import itertools 


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    pass


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """

    x = T[0, -1]
    y = T[1, -1]
    z = T[2, -1]

    phi = -np.arctan2(T[2,1], T[2,2])
    
    pose = np.array([x, y, z, phi])
    return pose

def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """

    s = np.array([[0, -w[2], w[1], v[0]],[w[2], 0, -w[0], v[1]],[-w[1], w[0], 0, v[2]], [0, 0, 0, 0]])
    
    return s

# def get_rot_matrix(omega, joint_angle):
#     R = np.identity(3) + np.sin(joint_angle) * s[0:2, 0:2] + (1 - np.cos(joint_angle))* np.dot(s[0:2, 0:2],s[0:2, 0:2])
#     return R

def get_pox(twist, joint_angle):
    s = to_s_matrix(twist[3:6], twist[0:3])
    # R = np.identity(3) + np.sin(joint_angle) * s[0:2, 0:2] + (1 - np.cos(joint_angle))* np.dot(s[0:2, 0:2],s[0:2, 0:2])
    # P = np.dot(np.identity(3)*joint_angle + (1 - np.cos(joint_angle) * s[0:2, 0:2]) + (joint_angle - np.sin(joint_angle)* np.dot(s[0:2, 0:2],s[0:2, 0:2]) ), s[0:2, -1])
    
    return expm(joint_angle * s)
    # return np.vstack((np.hstack((R, P)), [0, 0, 0, 1]))

def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      join t_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    g_st = np.eye(4)

    for joint_angle, twist in zip(joint_angles, s_lst):
        # s = to_s_matrix(twist[3:5], twist[0:2])
        # R = np.identity(3) + np.sin(joint_angle) * s[0:2, 0:2] + (1 - np.cos(joint_angle))* np.dot(s[0:2, 0:2],s[0:2, 0:2])
        # P = np.dot(np.identity(3)*joint_angle + (1 - np.cos(joint_angle) * s[0:2, 0:2]) + (joint_angle - np.sin(joint_angle)* np.dot(s[0:2, 0:2],s[0:2, 0:2]) ), s[0:2, -1])

        g_st = np.dot(g_st, get_pox(twist, joint_angle))

    g_st = np.dot(g_st, m_mat)

    return g_st

def get_gst(phi, psi, p):
    # phi: theta1 (first joint angle)
    # psi: horizontal tilt

    rot_x = np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)], [0, np.sin(psi), np.cos(psi)]])
    rot_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    R = np.dot(rot_z, rot_x)
    # phi = -np.arctan2(R[2,1], R[2,2])
    # print("phi = " + str(phi))
    gst = np.vstack((np.column_stack((R, p)), np.array([0, 0, 0, 1])))
    return gst

def find_goal_pose_for_RRR(pose, theta1):
    # parameters
    l1, l2, l3, l4, l5 = 103.91, 200.0, 50.0, 200.0, 174.15

    phi = pose[3] # block orientaiton
    psis = np.array([pose[4], -np.pi/4, 0.0, -np.pi/2, np.pi/4, np.pi/2]) # horizontal tilt
    
    for psi in psis:
        # desired g_st
        gd = get_gst(theta1, psi, pose[0:3])
        Rd = gd[0:3, 0:3]
        pd = gd[0:3, -1]
        
        p_shoulder = np.array([0, 0, l1])
        p_wrist = pd - np.dot(Rd, np.array([0, l5, 0])) - p_shoulder
 
        x_goal = np.sqrt(p_wrist[0]**2 + p_wrist[1]**2)
        y_goal = p_wrist[2]

        l_sq = l2**2 + l3**2
        l = np.sqrt(l_sq)

        r_sq = x_goal**2 + y_goal**2
        r = np.sqrt(r_sq)
        if (l + l4 >= r):
            goal = np.array([x_goal, y_goal, l_sq, l, r_sq, r, psi])
            return goal

    return None

def IK_geometric(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    IK_sol = np.empty((0,5))
    # parameters
    l1, l2, l3, l4, l5 = 103.91, 200.0, 50.0, 200.0, 174.15
    phi =  pose[3]
    psi = pose[4]

    # First find theta1 and theta5
    theta1 = -np.arctan2(pose[0], pose[1])
    theta1 = clamp(theta1)

    # Now reduced to planar RRR manipulator
    # define p_shoulder and p_wrist on joint 2 and 4, then normaliez the z w.r.t. the shoulder

    goal = find_goal_pose_for_RRR(pose, theta1)

    if goal is None:
        return None
    else:
        x_goal, y_goal, l_sq, l, r_sq, r, psi = goal[0], goal[1], goal[2], goal[3], goal[4], goal[5], goal[6]
    
    if psi != -np.pi/2:
        theta5 = 0
    else:
        theta5 = theta1 - pose[3]

    # print("x goal: " + str(x_goal) + " y goal: " + str(y_goal + l1))
    alpha = np.arctan2(l3, l2)
    beta = np.arctan2(y_goal, x_goal)
    c3 = (r_sq - l_sq - l4**2) / (2*l*l4)
    ac3 = np.arccos(c3)

    theta3 = np.array([ac3, -ac3])
    # print("Elbow down theta3: " + str(theta3[0]* 180/np.pi) + "elbow up theta3: " + str(theta3[1]* 180/np.pi))
    q3 = np.array([clamp(-np.pi/2 + alpha - theta3[0]), clamp(-np.pi/2 + alpha - theta3[1])])

    q2 = np.empty(2)
    q4 = np.empty(2)
    for i in range(theta3.size):
        gamma = np.arctan2(l4*np.sin(theta3[i]), l+l4 * np.cos(theta3[i]))
        theta2 = beta - gamma
        q4[i] = clamp(-psi + theta3[i] + theta2)
        q2[i] = clamp(np.pi/2 - alpha - theta2)


    IK_sol = np.array([[theta1, q2[0], q3[0], q4[0], theta5], 
                        [theta1, q2[1], q3[1], q4[1], theta5], 
                        [theta1, q2[0], q3[0], q4[0], clamp(theta5 - np.pi)], 
                        [theta1, q2[1], q3[1], q4[1], clamp(theta5 - np.pi)]])
    return IK_sol, psi

def get_IK_joint_angles(pose):

    # place the arm above the block
    first_pose = np.array([pose[0], pose[1], pose[2]+100, pose[3], pose[4]])
    # print("first_pose = " + str(first_pose))
    first_IK_sol, psi = IK_geometric(first_pose)
    # print("first_ik_sol = " + str(first_IK_sol))
    new_pose = np.array([pose[0], pose[1], pose[2], pose[3], psi])
    # pose[4] = psi
    if first_IK_sol is None:
        return None
    
    # move arm to grab the block
    else:
        final_IK_sol, psi = IK_geometric(new_pose)
    
        sol = np.array([first_IK_sol[1], final_IK_sol[1]])
        return sol, psi


def sp1(xi, p, q):
    # 1 solution
    v = xi[0:3]
    w = xi[3:7]

    w_norm = np.linalg.norm(w)
    r = np.cross(w, v) / (w_norm**2)

    u = p - r
    v = q - r

    up = u - np.dot(w, np.dot(w.T, u))
    vp = v - np.dot(w, np.dot(w.T, v))

    theta = np.arctan2(np.dot(w.T, np.cross(up, vp)), np.dot(up.T, vp))

    return theta

def find_intersection(w1, w2, r1, r2):
    c = r1
    e = w1
    d = r2
    f = w2
    g = d - c
    fcg = np.cross(f, g)
    fce = np.cross(f, e)
    if (np.linalg.norm(fce) == 0):
        return np.array([np.inf, np.inf, np.inf])
    elif (np.dot(fcg.T, fce) >= 0):
        return c + np.linalg.norm(fcg) / np.linalg.norm(fce) * e
    else:
        return c - np.linalg.norm(fcg) / np.linalg.norm(fce) * e

def sp2(xi1, xi2, p, q, r):
    # Two, one, or no solutions

    sol = None

    v1 = xi1[0:3]
    w1 = xi1[3:7]
    v2 = xi2[0:3]
    w2 = xi2[3:7]

    w1_norm = np.linalg.norm(w1)
    w2_norm = np.linalg.norm(w2)

    r1 = np.cross(w1, v1) / (w1_norm**2)
    r2 = np.cross(w2, v2) / (w2_norm**2)

    # r = find_intersection(w1, w2, r1, r2)

    # if (np.linalg.norm(r) == np.inf):
    #     return None
    u = p - r
    v = q - r

    w1w2 = np.dot(w1.T, w2)
    w1cw2 = np.cross(w1, w2)
    w2u = np.dot(w2.T, u)
    w1v = np.dot(w1.T, v)
    # alpha = (w1w2*w2u - w1v) / (w1w2**2 - 1)
    # beta = (w1w2*w1v - w2u) / (w1w2**2 - 1)
    alpha = (np.dot(w1w2*w2.T, u) - w1v) / (w1w2**2 - 1)
    beta = (np.dot(w1w2*w1.T, v)- w2u) / (w1w2**2 - 1)
    gamma_sq = (np.linalg.norm(u)**2 - alpha**2 - beta**2 - 2*alpha*beta*w1w2) / (np.linalg.norm(w1cw2)**2)

    if gamma_sq == 0:
        # one solution
        z = alpha*w1 + beta*w2
        c = z + r
        theta1 = sp1(-xi1, q, c)
        theta2 = sp1(xi2, p, c)
        sol = np.array([[theta1], [theta2]])
    else:
        z1 = alpha*w1 + beta*w2 + np.sqrt(gamma_sq)*w1cw2
        z2 = alpha*w1 + beta*w2 - np.sqrt(gamma_sq)*w1cw2
        c1 = z1 + r
        c2 = z2 + r
        theta11 = sp1(-xi1, q, c1)
        theta21 = sp1(xi2, p, c1)
        theta12 = sp1(-xi1, q, c2)
        theta22 = sp1(xi2, p, c2)
        sol = np.array([[theta11, theta12], [theta21, theta22]])
    return sol

def sp3(xi, p , q, delta):
    # two, one, or no solutions
    v = xi[0:3]
    w = xi[3:7]
    w_norm = np.linalg.norm(w)
    r = np.cross(w, v) / (w_norm**2)
    u = p - r
    v = q - r
    up = u - np.dot(w, np.dot(w.T, u))
    vp = v - np.dot(w, np.dot(w.T, v))
    theta_0 = np.arctan2(np.dot(w.T, np.cross(up, vp)), np.dot(up.T, vp))
    delta_prime_sq = delta**2 - np.abs(np.dot(w.T, p-q))**2

    u_prime_norm = np.linalg.norm(up)
    v_prime_norm = np.linalg.norm(vp)

    frac = (u_prime_norm**2 + v_prime_norm**2 - delta_prime_sq) / (2 * u_prime_norm * v_prime_norm)

    if frac > 1 and frac < -1:
        # no solution
        return None
    elif frac == 1 or frac == -1:
        # one solution
        return theta_0
    else:
        # two solutions
        theta = np.arccos(frac)
        theta1 = theta_0 + theta
        theta2 = theta_0 - theta
        return np.array([theta1 , theta2])

def IK_padenkahan(s_lst, gst0, pose):
    # parameters
    l1 = 103.91
    l2 = 200.0
    l3 = 50
    l4 = 200
    l5 = 174.15

    xi1 = s_lst[0]
    xi2 = s_lst[1]
    xi3 = s_lst[2]
    xi4 = s_lst[3]
    xi5 = s_lst[4]

    # g_st0 inverse
    R0 = gst0[0:3, 0:3]
    p0 = gst0[0:3, -1]
    gst0_inv = np.vstack((np.column_stack((R0.T, -np.dot(R0.T, p0))), np.array([0, 0, 0, 1])))

    # desired g_st
    gd = get_gst(pose[3], pose[4], pose[0:3])
    Rd = gd[0:3, 0:3]
    pd = gd[0:3, -1]

    IK_sol = np.empty((0,5))

    # q1 = theta1 
    theta1 = np.arctan2(-Rd[0,1], Rd[1,1])
    theta1 = clamp(theta1)
    # Solving for theta2 and theta 3

    p45 = pd - np.dot(Rd, np.array([0, l5, 0])) # double check the calculation
    g1d = np.dot(get_pox(-xi1, theta1), gd)
    g1 = np.dot(g1d , gst0_inv)
    
    # define p2 on joint 2
    p2 = np.array([0, 0, l1])
    g1p45 = np.dot(g1, np.append(p45, 0))
    delta = np.linalg.norm(g1p45[0:3] - p2)

    # Use SP3 to solve for theta3 (two solutions)
    theta3s = sp3(xi3, p45 , p2, delta)

    # Use SP1 to solve for theta2 (one solutions)
    # q2 = None
    # p3 = np.array([0, l3, l1+l2])
    p3 = p2
    for theta3 in theta3s:
        theta3 = clamp(theta3)
        p345 = np.dot(get_pox(xi3, theta3), np.append(p45, 0))
        theta2 = sp1(xi2, p345[0:3], g1p45[0:3])
        # np.append(q2, theta2)
    
    # Use SP2 to solve for theta4, theta5 (None, one, or two solutions)
        g32 = np.dot(get_pox(-xi3, theta3), get_pox(-xi2, theta2))
        g2 = np.dot(g32, g1)
        q3 = np.dot(g2, np.append(p3, 0))
        
        sol = sp2(xi4, xi5, p3, q3[0:3], p45)
        print(sol.shape[1])
        if sol == None:
            return None
        elif sol.shape[1] == 1:
            theta4 = clamp(sol[0])
            theta5 = clamp(sol[1])
            q = np.array([[theta1, theta2, theta3, theta4, theta5]])
            IK_sol = np.append(IK_sol, q, axis = 0)
        elif sol.shape[1] == 2:
            theta41 = clamp(sol[0,0])
            theta51 = clamp(sol[1,0])
            theta42 = clamp(sol[0,1])
            theta52 = clamp(sol[1,1])
            q = np.array([[theta1, theta2, theta3, theta41, theta51], 
                          [theta1, theta2, theta3, theta42, theta52]])
            IK_sol = np.append(IK_sol, q, axis = 0)
    return IK_sol
