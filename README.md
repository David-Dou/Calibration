# Fringe projection measurement system calibration
Calibration mainly realizes the positioning of solid circle identification points, and the calibration functions of bilateral telecentric cameras and projectors.
The calibration function is implemented in models/camera_calibration and dlp_calibration, the circle center is extracted in utils/extract_circle_center, the main realizes the calibration of the measurement system, run_with_rebuild is used as a test case, and the demo test result is obtained as

![demo](/demo/demo_2D_heatmap.png "2D_heatmap")
![demo](/demo/demo_3D_heatmap.png "3D_heatmap")

# Usage-Calibrate then rebuild
The calibration experiment of the grating projection measurement system can be carried out using the calibration program, and verified by reconstruction, first clone the repository

    git clone https://github.com/David-Dou/Calibration.git

then use

    python main.py --camera_img_width 4096 --camera_img_height 3072 --dlp_img_width 912 --dlp_img_height 1140 

for system calibration.
during rebuilding, use

    python run_with_rebuild.py 
    --camera_img_width 4096 --camera_img_height 3072 --dlp_img_width 912 --dlp_img_height 1140 
    --use_calibrated_params False 

# Usage-Rebuild
Tests can be directly reconstructed using the calibrated model, i.e. directly using

    python run_with_rebuild.py 
    --camera_img_width 4096 --camera_img_height 3072 --dlp_img_width 912 --dlp_img_height 1140 
    --use_calibrated_params True