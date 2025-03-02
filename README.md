# BFMC
# Bosch Future Mobility Challenge (BFMC) 2025 - README

## Overview
The Bosch Future Mobility Challenge (BFMC) is a competition where teams develop autonomous vehicles capable of navigating a scaled-down city environment. The challenge includes lane following, ramp climbing, traffic sign detection, pedestrian avoidance, obstacle handling, and overall autonomous decision-making.

This README file serves as a guide to the essential challenges and milestones required to complete and optimize your BFMC 2025 autonomous vehicle system.

---

## **Challenges & Tasks**

### **1. Manual Control Development**
- Implement a system for **manual remote control** via keyboard, joystick, or mobile app.
- Ensure real-time communication between the remote control and the vehicle using **UDP**.
- Optimize **steering response** and **speed control** for precise handling.

### **2. Autonomous Lane Following**
- Develop and optimize **lane detection** using a CNN-based or edge ML model.
- Implement **steering control** to maintain the car at the center of the lane.
- Ensure stable movement at **constant speed**.
- Improve accuracy under different lighting conditions.

### **3. High-Speed Driving & Ramp Climbing**
- Tune **PID control** for accurate high-speed stability.
- Implement **ramp detection** and adjust acceleration accordingly.
- Optimize torque distribution for smooth ramp climbing and descent.

### **4. Traffic Sign & Light Detection**
- Implement **traffic sign recognition** using an edge ML model.
- Detect and interpret **traffic lights** via UDP Wi-Fi communication.
- Develop a decision-making system for **stopping, yielding, and turning**.

### **5. Pedestrian & Obstacle Avoidance**
- Detect pedestrians and obstacles using **monocular depth estimation**.
- Implement **collision avoidance algorithms** to safely navigate around obstacles.
- Optimize **reaction time** and vehicle stopping distance.

### **6. Roundabout & Intersection Handling**
- Implement **roundabout entry and exit logic**.
- Develop an efficient way to handle **various intersections**.
- Prioritize movements based on **traffic rules and sign detection**.

### **7. Parking System**
- Develop an **autonomous parking algorithm**.
- Implement **parallel and perpendicular parking maneuvers**.
- Use **image processing** to detect parking zones.

### **8. Mapping & Localization**
- Implement a **track mapping system** to visualize the car's real-time position.
- Use traffic signs and road features for **position estimation**.
- Develop a **Google Maps-style interface** to track the car’s location.

### **9. Reinforcement Learning for Path Optimization**
- Train an RL model to **optimize lane keeping and obstacle avoidance**.
- Implement a policy to **improve driving performance over time**.
- Run RL models efficiently on **laptop + Raspberry Pi setup**.

### **10. ROS Integration**
- Fully integrate **ROS** for modular control and sensor data processing.
- Ensure **smooth communication** between Raspberry Pi, Nucleo, and Laptop.
- Use ROS for **data logging and real-time visualization**.

### **11. System Performance & Optimization**
- Ensure minimal **latency** between perception and control systems.
- Optimize **code efficiency** for real-time performance.
- Improve power management for **maximum runtime**.

---

## **Hardware & Software Stack**
- **Hardware:** Raspberry Pi + RPi Camera v3, Nucleo-F401RE (mbed OS), BLDC Motor, Steering Servo.
- **Software:** Python (ML, Image Processing), C++ (Depth Estimation), ROS (Middleware), Simulink (Simulations).
- **Communication:** UDP for real-time data transfer.

---

## **Milestone Plan**
1. **Jan-Feb 2025:** Manual Control & Lane Following
2. **March 2025:** High-Speed Driving, Ramp Climbing
3. **April 2025:** Traffic Sign & Light Detection, Obstacle Avoidance
4. **May 2025:** Roundabout, Intersection Handling
5. **June 2025:** Parking System, Localization & Mapping
6. **July 2025:** Reinforcement Learning & Path Optimization
7. **Aug 2025:** System Optimization & Final Testing

---

## **Conclusion**
This README provides an overview of the core challenges and goals for BFMC 2025. The objective is to build an efficient, robust, and high-performing autonomous vehicle capable of handling all competition scenarios. Focus on optimization and real-time performance to maximize your chances of winning BFMC 2025!

---

**For further details, refer to the official BFMC documentation and competition rules.**

