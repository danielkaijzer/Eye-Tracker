# Eye Tracker project

A high-precision, low-latency eye tracker prototype. Use cases include medical, marketing, sports performance coaching, gaming, day-to-day life, etc.

# (Super) Team: 
Cody Lam, Daniel Kaijzer, Ethan Shim, Harwin He, Roselio Ortega

# Deliverables and milestones 
1. Physical Prototype with basic eye tracking (model-based), processing done on laptop/PC. Data streaming to the terminal. Milestone #1
2. A nice web app to visualize data being streamed in. Milestone #2
3. Run ML model inference on the prototype for improved performance. Milestone #3
4. Mobile eye tracking setup using Raspberry Pi streaming data over WiFi or BLE to a laptop. Milestone #4
5. Finished product with AI inference running on the Raspberry Pi and data streaming to our web app via WiFi. Milestone #5


```mermaid
graph TD
    subgraph Hardware Layer ["Hardware Layer (Wearable Glasses)"]
        A[120Hz Internal IR Camera] -->|Raw Byte Stream| C[Raspberry Pi / Laptop]
        B[30Hz Front-Facing Scene Cam] -->|Video Feed| C
    end

    subgraph Backend Layer ["Backend (C++ & Python)"]
        C --> D{Data Router}
        D -->|UDP Stream| E[C++ Engine]
        E -->|Geometric Processing| F[OpenCV Pupil Detection]
        
        D -->|Data Collation| G[Python ML Module]
        G -->|Inference| H[TensorFlow/Edge AI Model]
        
        F --> I[Gaze Vector Calculation]
        H --> I
    end

    subgraph Frontend Layer ["UI/UX (React & JS)"]
        I -->|WebSocket/Data Stream| J[React Dashboard]
        J --> K[Real-time Gaze Overlay]
        J --> L[Calibration Suite]
        J --> M[Performance Analytics]
    end

    subgraph Output ["Deliverables"]
        K --> N[Scene Video + Red Dot]
        M --> O[ML vs Geometric Comparison]
    end
```
