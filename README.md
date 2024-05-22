# vPull

## Overview

vPull is a Python project that implements an analysis of the clinical pull test for testing postural instability in Parkinson's disease. This analysis is based on camera-based markerless 3D tracking of the examined patient.

## Features

- Step detection based on keypoint velocity.
- Pull detection based on acceleration thresholds.
- Bending detection based on changes in the angle of keypoints.
- Configurable filter properties for preprocessing keypoint data.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/vPull.git
    cd vPull
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your data**:
    - Ensure you have the properties file (`vPull_properties.json`) and the sample keypoint data file (`sample_data.json`) in the project directory.

2. **Run the main script**:
    ```sh
    python main.py
    ```

## File Structure

- `main.py`: Main script to run the analysis.
- `vPull_utils.py`: Utility functions for the analysis.
- `vPull_properties.json`: Properties for the analysis including filter settings and thresholds.
- `sample_data.json`: Sample keypoint data for testing.

## Configuration

### `vPull_properties.json`
This file contains the configuration for the analysis:

```json
{
    "step_keypoints": ["left_ankle", "right_ankle"],
    "step_vel_thr": 0.7,
    "step_len_thr": 0.05,
    "step_interval_thr": 3,
    "pull_keypoints": ["left_shoulder", "right_shoulder"],
    "pull_baseline": 1.0,
    "pull_thr": 3,
    "bending_keypoints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "bending_thr": 0.25,
    "filter_properties": {
        "cutoff": 10,
        "order": 2,
        "gap_thr": 1
    }
}
```

### `sample_data.json`
This file contains the sample keypoint data structured as a 3D array with the following dimensions:
- Number of keypoints
- Each keypoint contains: timestamp, x, y, z, keypoint_confidence
- Number of frames

## Example Output

The analysis script (`main.py`) will output a dictionary containing the following metrics:
```json
{
    "pull_onset": 1.2,
    "pull_magnitude": 2.5,
    "1st_step_latency": 0.8,
    "1st_step_duration": 0.5,
    "1st_step_length": 0.3,
    "step_number": 3,
    "bending_amplitude": 10.5,
    "bending_latency": 0.7,
    "recovered": true,
    "recover_latency": 1.5
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## Acknowledgements

- This project uses the `numpy`, `pandas`, and `scipy` libraries for numerical computations and signal processing.
- Special thanks to all contributors and the open-source community for their valuable tools and resources.

## Contact

For any questions or inquiries, please contact [Max Wühr](mailto:max.wuehr@med.uni-muenchen.de).
```
