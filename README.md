# Introduction
- This is an electrophysiological heart simulator written in Python.  
- It can simulate patient-specific focal and rotor arrhythmias. And simulate fibrillations.  
- It computes action potentials and electrograms.  
- The code runs on Nvidia GPU for fast parallel computing.  
- For solving the heart model equations, 4th-order Runge–Kutta is implemented for the reaction part, and Crank-Nicolson is implemented for the diffusion part. 

# Contributors
- Jiyue He -- Owner and the main contributor. Jiyue He (Jay) received his PhD from the University of Pennsylvania, where he was honored with the student recognition award. As of 2026, He is a Postdoctoral Scholar at the University of California, San Francisco. His research includes artificial intelligence, numerical modeling, algorithm development, signal processing and data analysis.
- Mason Manetta -- Implemented mesh processing to automatically correct defects and smooth the surface while preserving overall geometry. As of 2026, Manetta is a Bioengineering PhD student at the University of California, Berkeley and the University of California, San Francisco. His research includes wearable medical devices, electrical impedance tomography, sleep genetics, and cardiac physiology.
- Jan Christoph -- Financial support. As of 2026, Christoph is an Assistant Professor at the University of California, San Francisco, and head of the Cardiac Vision Laboratory. He is a faculty member of the Cardiovascular Research Institute, with appointments in the Division of Cardiology, School of Medicine, and the Department of Bioengineering and Therapeutic Sciences. His research interests include cardiac electrophysiology and biomechanics, cardiac arrhythmia mechanisms, the physics of complex biological systems, artificial intelligence, numerical modeling and imaging.

# Instruction
- Run heart_sim_individual.py to compute one heart simulation. 
- Run heart_sim_batch.py to compute multiple heart simulations.  

# Examples of simulations
## Atrial fibrillation of a patient's left atrium
<img src="demo/2_23403.gif" alt="Demo" width="400" />  
<img src="demo/2_23403_AP_and_EGM.png" alt="Demo" width="400" />  

## Rotor arrhythmia of a patient's left atrium
<img src="demo/1_35000.gif" alt="Demo" width="400" />  
<img src="demo/1_35000_AP_and_EGM.png" alt="Demo" width="400" />  

## Focal arrhythmia of a patient's left atrium
<img src="demo/0_23403.gif" alt="Demo" width="400" />  
<img src="demo/0_23403_AP_and_EGM.png" alt="Demo" width="400" />  
