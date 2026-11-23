# Railway Freight Wheelset Overhaul Workshop (1:25 Simulation Twin)

A high-fidelity 1:25 physical digital twin and simulation environment for an automated railway freight wheelset maintenance and overhaul workshop, built with ROS 2 and Gazebo.

## Overview

The simulation environment models the complete industrial lifecycle of railway freight car wheelset overhaul operations across multiple specialized functional cells:

1. **Infeed & Identification**: Inbound wheelset unloading, optical identification, RFID tag verification, and initial dimensional inspection.
2. **Disassembly & Demounting**: Bearing front-end cap removal, axle-end bolts demounting, and protective ring extraction.
3. **Cleaning & Surface Preparation**: Surface washing, wiping, and pre-inspection degreasing.
4. **Non-Destructive Testing (NDT)**: Ultrasonic flaw detection and magnetic particle testing (MT/UT) hoods and scanning benches.
5. **Machining & Turning**: Wheel tread CNC re-profiling lathe cells and axle finishing stations.
6. **Bearing Press-Mounting**: Cold/hydraulic press cells for precision bearing mounting with real-time pressure-displacement curve tracking.
7. **Painting & Finishing**: Anti-corrosion coating spray portals, exterior finish inspection, and drying buffers.
8. **Storage & Material Logistics**: Finished wheelset storage gantries, stackers, buffer bays, and autonomous mobile robot (AMR / Scout Mini) material supply lines.

## Directory Structure

```text
simulation/wheelset_workshop/
├── worlds/
│   └── wheelset_workshop.world        # Master Gazebo simulation world (complete 1:25 workshop layout)
├── agv_network_01/                    # AMR navigation topological road network (Phase 1)
│   └── scene_layout.json
├── agv_network_02/                    # AMR navigation topological road network (Phase 2)
│   └── scene_layout.json
└── cameras.json                       # Camera intrinsic and extrinsic pose parameters
```

## Running the Simulation

### Prerequisites
- Ubuntu 22.04 LTS (Jammy Jellyfish)
- ROS 2 Humble Hawksbill
- Gazebo 11 (Gazebo Classic)

### Launching the Workshop World
```bash
# Launch master workshop simulation twin
gazebo simulation/wheelset_workshop/worlds/wheelset_workshop.world
```

### AMR Navigation & Material Supply
The autonomous logistics fleet is powered by the `amr_sched` core decision engine and communicates with the Gazebo simulation twin via ROS 2 topics and C-ABI FFI bridges. See the root [README.md](../../README.md) for full architecture and execution details.
