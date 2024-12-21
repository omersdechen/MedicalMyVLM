import numpy as np

class CyclicShift:
    def __init__(self, shift_x=0, shift_y=0):
        """
        Initialize the CyclicShift augmentation with desired shifts.
        Parameters:
        - shift_x: Number of pixels to shift horizontally (positive for right, negative for left).
        - shift_y: Number of pixels to shift vertically (positive for down, negative for up).
        """
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, image):
        """
        Make the CyclicShift callable so it integrates with augmentation pipelines.
        Parameters:
        - image: Input image as a NumPy array.
        Returns:
        - A dictionary with the augmented image under the key 'image'.
        """
        shift_x = self.shift_x % image.shape[1]
        shift_y = self.shift_y % image.shape[0]
        
        # Apply horizontal cyclic shift
        if shift_x != 0:
            image = np.roll(image, shift_x, axis=1)
        
        # Apply vertical cyclic shift
        if shift_y != 0:
            image = np.roll(image, shift_y, axis=0)
        
        # Return the result in a dictionary
        return {'image': image}
