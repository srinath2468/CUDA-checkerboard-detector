# CUDA-checkerboard-detector
 
 This checkerboard detector was created to work on the jetson TX2 that utilises CUDA. To use, download a checkerboard with ample spacing on all 4 sides so that it does not break the detection. We utilise a corner detector and checkerboards that are flushed to the sides will not be detected. This issue will be addressed soon.
 
 ---
 
 TODO:
 
 a. Make this a headers only library
 b. Solve the flush issue
 
 ---
 
 Usage:
 
 1. git clone "Repository"
 2. cd "Repository"
 3. cmake ..
 4. make -jn #n = number of cores
 5. Initialise the detector
 6. Send the image
 7. Release the detector


Output: 

---
checkerboard.Detect returns a vector of ordered points from top left to bottom right corners.


---

9x7 checkerboard

![result_1](https://user-images.githubusercontent.com/25114497/131876495-7d003f01-f3fa-4ae4-a256-1e1ede064078.jpg)

---

9x6 checkerboard

![result_2](https://user-images.githubusercontent.com/25114497/131876743-805ac7d5-2b4b-4044-8d2f-9fa431e9d40f.jpg)

--- 
7x9 checkerboard

![result_3](https://user-images.githubusercontent.com/25114497/131876826-ef243245-ff77-487d-8612-ad2b14e7c8ed.jpg)





 
