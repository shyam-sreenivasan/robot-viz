# UR5 Collaborative Robot: Kinematic Analysis and Trajectory Planning

A comprehensive kinematic framework for the Universal Robots UR5 manipulator, featuring forward kinematics, differential inverse kinematics, and real-time trajectory planning with interactive web-based visualization.

## Overview

This project implements a complete kinematic control system for the UR5 6-DOF collaborative robot, achieving smooth trajectory tracking with sub-3mm positional accuracy. The system includes forward kinematics using Denavit-Hartenberg parameters and a differential inverse kinematics controller that computes joint velocities in real-time using Jacobian pseudo-inverse methods.

## Key Features

- **Forward Kinematics**: DH parameter-based transformations and Product of Exponentials formulation
- **Differential Inverse Kinematics**: Velocity-level controller using damped least-squares pseudo-inverse for numerical stability near singularities
- **Trajectory Planning**: Three parametric motion primitives:
  - Circular paths (100mm radius)
  - Square paths (100mm sides)
  - Sine wave trajectories (80-150mm wavelength)
- **Interactive Visualization**: Web-based system built with Three.js and React featuring:
  - Real-time end-effector pose display
  - Joint angle control via sliders
  - Manipulability metrics
  - Animated waypoint visualization
  - 20Hz real-time control updates

## Performance

- Positional accuracy: <3mm across all trajectory types
- Circular trajectory error: ~1.5mm (highest accuracy)
- Square trajectory error: ~2-3mm (corners)
- Update frequency: 20Hz (50ms intervals)
- Workspace validation: Automatic verification within 900mm reachable envelope

## Technical Approach

The project evolved from an initial Newton-Raphson iterative solver (which encountered configuration discontinuities and convergence failures) to a differential IK controller that reformulates the problem as real-time proportional control. This velocity-level approach eliminates convergence issues and enables smooth, continuous motion suitable for educational and research applications.

## Links

- **📄 Full Report**: [Project Report (PDF)](https://github.com/shyam-sreenivasan/robot-viz/blob/master/Project-Report.pdf)
- **🎥 Video Demo**: [YouTube](https://youtu.be/uaj2goZ3IgM)
- **🚀 Live Demo**: [Interactive Visualization](https://robot-viz-entl.vercel.app/)
- **💻 Source Code**: [GitHub Repository](https://github.com/shyam-sreenivasan/robot-viz/blob/master/src/App.jsx)

## Technologies

- Three.js for 3D visualization
- React for interactive UI
- Jacobian pseudo-inverse with damped least-squares regularization
- Denavit-Hartenberg parameter framework
- Product of Exponentials formulation

## Author

Shyam Sreenivasan  
ME5250 - Robot Mechanics and Control - Project 2.1  
Northeastern University, Boston, MA

---

*This project demonstrates practical application of differential kinematics for real-time robot control, bridging theoretical foundations with interactive visual simulation.*