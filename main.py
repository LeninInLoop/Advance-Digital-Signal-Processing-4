import numpy as np
from matplotlib import pyplot as plt
import os

# Set numpy to print numbers with more precision
np.set_printoptions(precision=10, suppress=True)

# --- Define output directory ---
RESULT_DIR = "Result"


class DataGenerator:
    """Generates sets of data for filtering."""

    @staticmethod
    def generate_random_data(size: int = 100) -> np.ndarray:
        """Generates a 1D numpy array of random floats between 0.0 and 1.0."""
        return np.random.rand(size)


class DataProcessor:
    """
    Applies various filtering and processing operations to data.
    All methods are static and do not share state.
    """

    @staticmethod
    def apply_h_n(data: np.ndarray) -> np.ndarray:
        """
        Applies the specific IIR filter H(z) to the input data.

        H(z) = ( (1/6)z^-1 ) / ( 1 - (5/6)z^-1 + (1/6)z^-2 )

        This corresponds to the difference equation:
        y[n] = (5/6)y[n-1] - (1/6)y[n-2] + (1/6)x[n-1]
        """
        N = data.shape[0]
        y = np.zeros(N)

        # Iterate from n = 0 to N-1
        for n in range(N):
            # --- Get past values, handling initial conditions ---
            y_nm1 = y[n - 1] if n >= 1 else 0.0
            y_nm2 = y[n - 2] if n >= 2 else 0.0
            x_nm1 = data[n - 1] if n >= 1 else 0.0

            # --- Apply the difference equation ---
            y[n] = (5 / 6) * y_nm1 - (1 / 6) * y_nm2 + (1 / 6) * x_nm1

        return y

    @staticmethod
    def apply_downsample(data: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsamples the data by an integer factor (also known as decimation).

        Keeps every 'factor'-th sample, starting from the first (index 0).
        Example: data[::5]
        """
        if factor <= 0:
            raise ValueError("Downsampling factor must be > 0")
        return data[::factor]


# ############################################################################
# EFFICIENT POLYPHASE SOLUTION
# ############################################################################

class PolyphaseFilter:
    """
    **************************************************************************
    * NOTE                                   *
    **************************************************************************
    * This class implements the EFFICIENT M=5 polyphase decomposition filters *
    * (E0 to E4) for the filter H(z) defined in DataProcessor.apply_h_n.     *
    * *
    * This polyphase structure is the "correct" way to build an efficient    *
    * decimator, as it allows you to filter at the *lower* sample rate.       *
    **************************************************************************

    Implements the M=5 polyphase decomposition filters (E0 to E4)
    for H(z) = 1/(1 - 0.5z^-1) - 1/(1 - (1/3)z^-1).

    This class uses the "Direct Form" implementation based on the
    combined second-order difference equations derived previously.
    """

    # Shared denominator coefficients (feedback)
    # y[n] = A1 * y[n-1] + A2 * y[n-2] + ...
    A1 = 275.0 / 7776.0
    A2 = -1.0 / 7776.0

    @staticmethod
    def _apply_filter(data: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """
        Generic helper function to apply a 2nd-order IIR filter.
        y[n] = A1*y[n-1] + A2*y[n-2] + b0*x[n] + b1*x[n-1]
        """
        N = data.shape[0]
        y = np.zeros(N)

        for n in range(N):
            # Get past output values (feedback)
            y_nm1 = y[n - 1] if n >= 1 else 0.0
            y_nm2 = y[n - 2] if n >= 2 else 0.0

            # Get current and past input values (feedforward)
            x_n = data[n]
            x_nm1 = data[n - 1] if n >= 1 else 0.0

            # Apply the full difference equation
            y[n] = (PolyphaseFilter.A1 * y_nm1) + \
                   (PolyphaseFilter.A2 * y_nm2) + \
                   (b0 * x_n) + \
                   (b1 * x_nm1)

        return y

    @staticmethod
    def apply_e0(data: np.ndarray) -> np.ndarray:
        """Applies the E0(z) polyphase filter."""
        B0 = 0.0
        B1 = 211.0 / 7776.0
        return PolyphaseFilter._apply_filter(data, B0, B1)

    @staticmethod
    def apply_e1(data: np.ndarray) -> np.ndarray:
        """Applies the E1(z) polyphase filter."""
        B0 = 1.0 / 6.0
        B1 = 227.0 / 7776.0
        return PolyphaseFilter._apply_filter(data, B0, B1)

    @staticmethod
    def apply_e2(data: np.ndarray) -> np.ndarray:
        """Applies the E2(z) polyphase filter."""
        B0 = 5.0 / 36.0
        B1 = 19.0 / 7776.0
        return PolyphaseFilter._apply_filter(data, B0, B1)

    @staticmethod
    def apply_e3(data: np.ndarray) -> np.ndarray:
        """Applies the E3(z) polyphase filter."""
        B0 = 19.0 / 216.0
        B1 = 5.0 / 7776.0
        return PolyphaseFilter._apply_filter(data, B0, B1)

    @staticmethod
    def apply_e4(data: np.ndarray) -> np.ndarray:
        """Applies the E4(z) polyphase filter."""
        B0 = 65.0 / 1296.0
        B1 = 1.0 / 7776.0
        return PolyphaseFilter._apply_filter(data, B0, B1)


# ############################################################################
# PLOTTING
# ############################################################################

class Plotter:
    """Handles all plotting operations."""

    @staticmethod
    def _plot_original(ax: plt.Axes, data: np.ndarray):
        """Plots the original data on a given axis."""
        ax.set_title("1. Original Data (x[n])")
        ax.stem(data)
        ax.grid(True)
        ax.set_xlim((-1, len(data)))
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Sample (n)")

    @staticmethod
    def _plot_processed(ax: plt.Axes, data: np.ndarray):
        """Plots the processed (filtered) data on a given axis."""
        ax.set_title("2. Processed Data (y[n] = x[n] * h[n])")
        ax.stem(data)
        ax.grid(True)
        ax.set_xlim((-1, len(data)))
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Sample (n)")

    @staticmethod
    def _plot_downsampled_original_axis(ax: plt.Axes, data: np.ndarray, factor: int):
        """Plots the downsampled data, but spaced out on the original time axis."""
        ax.set_title(f"3. Downsampled Data on Original Axis (y_ds[n] = y[n * {factor}])")

        n_original_axis = np.arange(len(data)) * factor
        n_limit = (len(data) - 1) * factor + 1  # Get full range

        ax.stem(n_original_axis, data)
        ax.grid(True)
        ax.set_xlim((-1, n_limit))
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Sample (n)")

    @staticmethod
    def _plot_downsampled_new_axis(ax: plt.Axes, data: np.ndarray):
        """Plots the downsampled data as its own new, dense sequence."""
        ax.set_title("4. Downsampled Data (New Sequence y_ds[m])")

        ax.stem(data)  # Plots against m = [0, 1, 2, ...]
        ax.grid(True)
        ax.set_xlim((-1, len(data)))
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Sample (m)")

    @staticmethod
    def plot_data(original: np.ndarray, processed: np.ndarray,
                  downsampled: np.ndarray, factor: int, save_path: str):
        """
        Creates a 4-panel figure showing the results of the
        filter-then-downsample operation and saves it.
        """
        # Create a figure with 4 subplots, stacked vertically
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        fig.suptitle("Inefficient Method: Filter-Then-Downsample", fontsize=16)

        # Call the helper function for each subplot
        Plotter._plot_original(axes[0], original)
        Plotter._plot_processed(axes[1], processed)
        Plotter._plot_downsampled_original_axis(axes[2], downsampled, factor)
        Plotter._plot_downsampled_new_axis(axes[3], downsampled)

        # Clean up layout, save the figure, and close it
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(save_path)
        plt.close(fig)

    @staticmethod
    def _plot_compared_stream(ax: plt.Axes, data: np.ndarray, title: str):
        """Helper for plotting a single stream in the comparison plot."""
        ax.set_title(title)
        ax.stem(data)
        ax.grid(True)
        ax.set_xlim((-1, len(data)))
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Sample (m)")

    @staticmethod
    def _plot_difference(ax: plt.Axes, data: np.ndarray, title: str):
        """Helper for plotting the difference stream."""
        ax.set_title(title)
        ax.stem(data)
        ax.grid(True)
        ax.set_xlim((-1, len(data)))
        ax.set_ylabel("Error")
        ax.set_xlabel("Sample (m)")

        # Get min/max for sensible y-limits
        min_val = np.min(data)
        max_val = np.max(data)
        # Add padding
        padding = (max_val - min_val) * 0.1 + 1e-9
        ax.set_ylim((min_val - padding, max_val + padding))

    @staticmethod
    def plot_comparison(y_inefficient: np.ndarray, y_efficient: np.ndarray,
                        n_plot: int, save_path: str):
        """
        Creates a 3-panel figure comparing the output of the inefficient
        and efficient polyphase methods and saves it.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Method Comparison: Inefficient vs. Efficient Polyphase", fontsize=16)

        # Get slices to plot
        y_ineff_slice = y_inefficient[:n_plot]
        y_eff_slice = y_efficient[:n_plot]

        Plotter._plot_compared_stream(axes[0], y_ineff_slice, "1. Inefficient Output (Filter-then-Downsample)")
        Plotter._plot_compared_stream(axes[1], y_eff_slice, "2. Efficient Output (Polyphase Decimator)")

        # Clean up layout, save the figure, and close it
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(save_path)
        plt.close(fig)

    @staticmethod
    def plot_polyphase_io_streams(x_streams: list, y_streams: list, n_plot: int, save_path: str):
        """
        Creates and saves a multi-panel plot showing the input (x_k) and output (y_k)
        for each polyphase stream.
        """
        M = len(x_streams)
        if M != len(y_streams):
            raise ValueError("Input and output stream lists must have the same length")

        # Create a figure with M rows and 2 columns
        # Increase figure width to accommodate two plots side-by-side
        fig, axes = plt.subplots(M, 2, figsize=(14, 3 * M), sharex=True)

        fig.suptitle("Polyphase Stream Inputs (x_k[m]) and Outputs (y_k[m])", fontsize=16)

        for k in range(M):
            # --- Plot Input Stream (Left Column) ---
            ax_in = axes[k, 0]
            x_slice = x_streams[k][:n_plot]

            title_in = f"Input Stream x_{k}[m]"
            if k == 0:
                title_in = "Input Streams (Left)\n" + title_in
            ax_in.set_title(title_in)
            ax_in.stem(x_slice)
            ax_in.grid(True)
            ax_in.set_ylabel("Amplitude")
            ax_in.set_xlim((-1, n_plot))

            # --- Plot Output Stream (Right Column) ---
            ax_out = axes[k, 1]
            y_slice = y_streams[k][:n_plot]

            title_out = f"Output Stream y_{k}[m] = x_{k}[m] * E_{k}[m]"
            if k == 0:
                title_out = "Output Streams (Right)\n" + title_out
            ax_out.set_title(title_out)
            ax_out.stem(y_slice)
            ax_out.grid(True)
            ax_out.set_ylabel("Amplitude")
            ax_out.set_xlim((-1, n_plot))

        # Add x-label only to the bottom-most plots
        axes[-1, 0].set_xlabel("Sample (m)")
        axes[-1, 1].set_xlabel("Sample (m)")

        # Clean up layout, save the figure, and close it
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust for suptitle
        plt.savefig(save_path)
        plt.close(fig)

    @staticmethod
    def plot_polyphase_decomposition(original_data: np.ndarray, x_streams: list, M: int, n_plot: int, save_path: str):
        """
        Creates and saves a plot showing the original signal x[n] and how it is
        decomposed into the M polyphase streams (x_k[m]), plotted on the
        original high-rate 'n' axis.
        """

        # Create a figure with M+1 subplots, stacked vertically
        fig, axes = plt.subplots(M + 1, 1, figsize=(14, 2.5 * (M + 1)), sharex=True)

        fig.suptitle(f"Polyphase Decomposition of Original Signal x[n] (M={M})", fontsize=16)

        # --- Plot Original x[n] (Top Plot) ---
        ax_orig = axes[0]
        ax_orig.stem(original_data)
        ax_orig.set_title("Original High-Rate Signal x[n]")
        ax_orig.set_ylabel("Amplitude")
        ax_orig.grid(True)

        # --- Loop and Plot Each Stream ---
        for k in range(M):
            ax_stream = axes[k + 1]
            stream_data = x_streams[k]

            # Calculate the 'n' indices for each 'm' sample in this stream
            # based on the definition: x_k[m] = x[mM - k]
            m_axis = np.arange(len(stream_data))
            n_axis = m_axis * M - k

            # Filter out indices that are out of our plotting range
            valid_mask = (n_axis >= 0) & (n_axis < n_plot)
            n_to_plot = n_axis[valid_mask]
            data_to_plot = stream_data[valid_mask]

            # Use a consistent color for the stream
            color = f'C{k + 1}'

            # Plot the stems
            ax_stream.stem(n_to_plot, data_to_plot, linefmt=f'{color}-', markerfmt=f'{color}o', basefmt=' ')

            ax_stream.set_title(f"Stream x_{k}[m] (samples from x[mM - {k}])")
            ax_stream.set_ylabel("Amplitude")
            ax_stream.grid(True)

        # Finalize Plot
        axes[-1].set_xlabel("Sample (n) - High-Rate Axis")
        axes[0].set_xlim((-1, n_plot))  # Set shared x-axis limit

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust for suptitle
        plt.savefig(save_path)
        plt.close(fig)

    # --- NEW METHOD (FOR LATEST REQUEST) ---
    @staticmethod
    def plot_polyphase_reconstruction(processed_data: np.ndarray, y_streams: list,
                                      y_efficient: np.ndarray, M: int, n_plot: int,
                                      save_path: str):
        """
        Creates and saves a plot showing how the low-rate polyphase output
        streams (y_k[m]) sum together to form the final downsampled signal,
        all plotted on the original high-rate 'n' axis.
        """

        # Create a figure with M+2 subplots
        fig, axes = plt.subplots(M + 2, 1, figsize=(14, 2.5 * (M + 2)), sharex=True)

        fig.suptitle(f"Polyphase Reconstruction of Output Signal (M={M})", fontsize=16)

        # --- Plot "Ground Truth" y[n] (Top Plot) ---
        ax_orig = axes[0]
        ax_orig.stem(processed_data, 'b')
        ax_orig.set_title("Original High-Rate Output y[n] (Ground Truth)")
        ax_orig.set_ylabel("Amplitude")
        ax_orig.grid(True)

        # --- Loop and Plot Each Output Stream y_k[m] ---
        for k in range(M):
            ax_stream = axes[k + 1]
            stream_data = y_streams[k]

            # Calculate the 'n' indices for each 'm' sample in this stream
            # All streams y_k[m] contribute to the output sample at n = mM
            m_axis = np.arange(len(stream_data))
            n_axis = m_axis * M

            # Filter out indices that are out of our plotting range
            valid_mask = (n_axis >= 0) & (n_axis < n_plot)
            n_to_plot = n_axis[valid_mask]
            data_to_plot = stream_data[valid_mask]

            # Use a consistent color
            color = f'C{k + 1}'

            # Plot the stems
            ax_stream.stem(n_to_plot, data_to_plot, linefmt=f'{color}-', markerfmt=f'{color}o', basefmt=' ')

            ax_stream.set_title(f"Stream Contribution y_{k}[m] (plotted at n=mM)")
            ax_stream.set_ylabel("Amplitude")
            ax_stream.grid(True)

        # --- Plot the Summed Output y_efficient[m] (Bottom Plot) ---
        ax_sum = axes[M + 1]
        stream_data = y_efficient

        m_axis = np.arange(len(stream_data))
        n_axis = m_axis * M

        # Filter out indices that are out of our plotting range
        valid_mask = (n_axis >= 0) & (n_axis < n_plot)
        n_to_plot = n_axis[valid_mask]
        data_to_plot = stream_data[valid_mask]

        # Plot the stems in red to stand out
        ax_sum.stem(n_to_plot, data_to_plot, linefmt='r-', markerfmt='ro', basefmt=' ')
        ax_sum.set_title("Final Summed Output y_ds[m] = sum(y_k[m]) (plotted at n=mM)")
        ax_sum.set_ylabel("Amplitude")
        ax_sum.grid(True)

        # Finalize Plot
        axes[-1].set_xlabel("Sample (n) - High-Rate Axis")
        axes[0].set_xlim((-1, n_plot))  # Set shared x-axis limit

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust for suptitle
        plt.savefig(save_path)
        plt.close(fig)


# ############################################################################
# MAIN EXECUTION
# ############################################################################

def main():
    """
    Main function to demonstrate and compare the INEFFICIENT
    "filter-then-downsample" method with the EFFICIENT
    polyphase decimator method.
    """
    # --- 0. Setup Output Directory ---
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        print(f"Created directory: {RESULT_DIR}")

    # --- 1. Configuration ---
    N_DATA = 1000  # Total number of data points to generate
    M = 5  # Downsampling (decimation) factor
    N_PLOT_ORIGINAL = 50  # Number of *original* samples to visualize

    # Calculate how many downsampled points correspond to N_PLOT_ORIGINAL
    N_PLOT_DOWNSAMPLED = int(np.ceil(N_PLOT_ORIGINAL / M))

    print(f"Comparing 'Filter-then-Downsample' with 'Efficient Polyphase' (M={M})")

    # --- 2. Generate Data ---
    data = DataGenerator.generate_random_data(N_DATA)
    print(f"Generated {N_DATA} random data points.")

    # --- 3. Process Data (Inefficient Method) ---
    print("\n--- Method 1: Inefficient (Filter-then-Downsample) ---")

    # First, filter the *entire* high-rate data
    print("Filtering at high sample rate (n)...")
    processed_data = DataProcessor.apply_h_n(data)

    # Second, throw away M-1 out of every M samples
    print(f"Downsampling by {M} (throwing away samples)...")
    y_inefficient = DataProcessor.apply_downsample(processed_data, factor=M)
    print("Inefficient processing complete.")

    # --- 4. Process Data (Efficient Polyphase Method) ---
    print("\n--- Method 2: Efficient (Polyphase Decimator) ---")

    # Calculate size of downsampled arrays
    N_ds = int(np.ceil(N_DATA / M))

    # De-interleave the input signal x[n] into 5 polyphase streams
    # x_k[m] = x[mM - k]
    print("De-interleaving input into 5 polyphase streams...")
    x_poly_streams = []
    for k in range(M):
        stream = np.zeros(N_ds)
        for m in range(N_ds):
            n = m * M - k
            if 0 <= n < N_DATA:
                stream[m] = data[n]
        x_poly_streams.append(stream)

    # --- 4b. PLOT POLYPHASE DECOMPOSITION (Plot 4) ---
    save_path_4 = os.path.join(RESULT_DIR, "4_polyphase_decomposition.png")
    Plotter.plot_polyphase_decomposition(
        original_data=data[:N_PLOT_ORIGINAL],
        x_streams=x_poly_streams,
        M=M,
        n_plot=N_PLOT_ORIGINAL,
        save_path=save_path_4
    )
    print(f"Saved polyphase decomposition plot to {save_path_4}")

    # Filter each stream at the low sample rate
    print("Filtering 5 streams at low sample rate (m)...")
    filter_funcs = [
        PolyphaseFilter.apply_e0,
        PolyphaseFilter.apply_e1,
        PolyphaseFilter.apply_e2,
        PolyphaseFilter.apply_e3,
        PolyphaseFilter.apply_e4
    ]

    y_poly_streams = []
    for k in range(M):
        y_poly_streams.append(filter_funcs[k](x_poly_streams[k]))

    # --- 4c. PLOT POLYPHASE I/O STREAMS (Plot 3) ---
    save_path_3 = os.path.join(RESULT_DIR, "3_polyphase_io_streams.png")
    Plotter.plot_polyphase_io_streams(
        x_streams=x_poly_streams,
        y_streams=y_poly_streams,
        n_plot=N_PLOT_DOWNSAMPLED,
        save_path=save_path_3
    )
    print(f"Saved polyphase I/O streams plot to {save_path_3}")

    # Combine (sum) the outputs
    print("Combining 5 filtered streams...")
    y_efficient = np.zeros(N_ds)
    for k in range(M):
        y_efficient += y_poly_streams[k]

    print("Efficient processing complete.")

    # --- 4d. PLOT POLYPHASE RECONSTRUCTION (NEW PLOT 5) ---
    save_path_5 = os.path.join(RESULT_DIR, "5_polyphase_reconstruction.png")
    Plotter.plot_polyphase_reconstruction(
        processed_data=processed_data[:N_PLOT_ORIGINAL],
        y_streams=y_poly_streams,
        y_efficient=y_efficient,
        M=M,
        n_plot=N_PLOT_ORIGINAL,
        save_path=save_path_5
    )
    print(f"Saved polyphase reconstruction plot to {save_path_5}")

    # --- 5. Compare Results ---
    print("\n--- Comparison ---")
    # Ensure arrays are same length for comparison
    min_len = min(len(y_inefficient), len(y_efficient))
    y_inefficient = y_inefficient[:min_len]
    y_efficient = y_efficient[:min_len]

    # --- 6. Plot Results ---
    print("\nPlotting and saving remaining results...")

    # Plot 1: Show the inefficient process (as before)
    save_path_1 = os.path.join(RESULT_DIR, "1_inefficient_process.png")
    Plotter.plot_data(
        original=data[:N_PLOT_ORIGINAL],
        processed=processed_data[:N_PLOT_ORIGINAL],
        downsampled=y_inefficient[:N_PLOT_DOWNSAMPLED],
        factor=M,
        save_path=save_path_1
    )
    print(f"Saved plot 1 to {save_path_1}")

    # Plot 2: Compare the two final outputs
    save_path_2 = os.path.join(RESULT_DIR, "2_method_comparison.png")
    Plotter.plot_comparison(
        y_inefficient=y_inefficient,
        y_efficient=y_efficient,
        n_plot=N_PLOT_DOWNSAMPLED,
        save_path=save_path_2
    )
    print(f"Saved plot 2 to {save_path_2}")


if __name__ == "__main__":
    main()