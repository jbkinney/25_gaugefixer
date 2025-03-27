import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches

def visualize_matrix(matrix, zero_threshold=1e-10, show_grid=True, figsize=(12, 8)):
    """
    Visualize a matrix with a continuous colormap and colorbar.
    Zero values are colored white, with diverging colors for positive and negative values.
    More intense colors for non-zero values with abrupt transition to white.
    
    Args:
        matrix: numpy array of values to visualize
        zero_threshold: values with absolute value below this threshold are treated as zero
        show_grid: whether to show the grid lines without tick marks (default: True)
        figsize: tuple of (width, height) for the figure size (default: (12, 8))
    """
    # Create a copy of the matrix where small values are set to exactly zero
    matrix_clean = np.copy(matrix)
    matrix_clean[np.abs(matrix) < zero_threshold] = 0.0
    
    # Find the maximum absolute value for symmetric color scaling
    vmax = np.max(np.abs(matrix_clean))
    if vmax == 0:
        vmax = 1.0  # Avoid division by zero if all values are zero
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Create a custom diverging colormap with a wider white region at center
    # Use modified start/end points to intensify colors and create abrupt transition
    
    # For red colors (negative values): Use a narrower range to get more intense reds
    colors_r = plt.cm.Reds_r(np.linspace(0.0, 0.8, 127))  
    
    # For blue colors (positive values): Use a narrower range to get more intense blues
    colors_b = plt.cm.Blues(np.linspace(0.2, 1.0, 127))   
    
    # Create multiple white points for a wider "zero" region in the colormap
    # This ensures values close to zero appear white
    white = np.array([1.0, 1.0, 1.0, 1.0])
    whites = np.array([white] * 2)  # Keep the 2 white points for the zero region
    
    # Stack the colors with the white region in the middle
    colors = np.vstack((colors_r, whites, colors_b))
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Use a two-slope normalization to ensure zero is exactly at the center
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot the matrix with the custom colormap
    im = plt.imshow(matrix_clean, cmap=custom_cmap, norm=norm, interpolation='nearest')
    plt.colorbar(im)
    
    # Get current axes
    ax = plt.gca()
    
    # Remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show_grid:
        # Add grid lines to show cell boundaries without tick marks
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(True, which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Make sure tick marks are not visible
        ax.tick_params(which='both', bottom=False, left=False, top=False, right=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example matrix
    example_matrix = np.array([[1.0, 0.25, 0.25],
                              [0.0, 0.75, -0.25],
                              [0.0, -0.25, 0.75]])
    visualize_matrix(example_matrix, show_grid=True) 