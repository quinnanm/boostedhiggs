import uproot
import matplotlib.pyplot as plt
import numpy as np

# Load the file
file_path = "fitDiagnosticsBlinded.root"  # Adjust the path if necessary
file = uproot.open(file_path)

# Navigate to the 'shapes_fit_b' directory
# shapes_fit_b = file["shapes_prefit"]
shapes_fit_b = file["shapes_fit_b"]

# Define control regions CR1 and CR2
regions = ["CR1", "CR2"]

# Background processes to plot and corresponding colors
background_processes = ["diboson", "ewkvjets", "fake", "singletop", "ttbar", "wjets", "wzqq", "zjets"]
colors = ["blue", "green", "cyan", "purple", "magenta", "red", "orange", "yellow"]

# Dictionary to store histograms for each region and process
histograms = {}

for region in regions:
    histograms[region] = {}
    # Retrieve all keys in the region and remove the ';1' suffix
    region_keys = {key.split(";")[0]: key for key in shapes_fit_b[region].keys()}
    for process in background_processes + ["data"]:
        try:
            # Access the object using the stripped key
            obj = shapes_fit_b[f"{region}/{region_keys[process]}"]
            # Check if the object is a TGraphAsymmErrors or TH1F and handle accordingly
            if obj.classname == "TGraphAsymmErrors":
                # Extract x (bin centers) and y (values) for TGraphAsymmErrors
                x_values = np.array(obj.member("fX"), dtype=np.float32)
                y_values = np.array(obj.member("fY"), dtype=np.float32)
            elif obj.classname == "TH1F":
                # For TH1F, extract the bin centers and values directly
                y_values = obj.values()
                bin_edges = obj.axes[0].edges()
                x_values = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers

            histograms[region][process] = (x_values, y_values)
        except KeyError:
            print(f"Process '{process}' not found in region '{region}'.")
        except Exception as e:
            print(f"Error accessing '{process}' in region '{region}': {e}")

# Plot stacked histograms for each region in linear scale
for region in regions:
    plt.figure(figsize=(10, 6))
    bin_centers = histograms[region][background_processes[0]][0]
    bin_width = bin_centers[1] - bin_centers[0]  # Assuming uniform bin width

    # Initialize an array to accumulate the stacked values
    cumulative_values = np.zeros_like(bin_centers)

    # Plot each background process stacked in linear scale
    for i, process in enumerate(background_processes):
        if process in histograms[region]:
            y_values = histograms[region][process][1]
            plt.bar(bin_centers, y_values, width=bin_width, bottom=cumulative_values,
                    color=colors[i % len(colors)], alpha=0.9, label=process, align='center')
            cumulative_values += y_values

    # Plot the data points as black markers with error bars, no connecting line
    if "data" in histograms[region]:
        plt.errorbar(histograms[region]["data"][0], histograms[region]["data"][1], 
                     yerr=np.sqrt(histograms[region]["data"][1]), fmt='o', color='black', label="Data")

    # Set the y-axis limits based on the region
    if region == "CR1":
        plt.ylim(0, 20)
    elif region == "CR2":
        plt.ylim(0, 70)

    plt.title(f"Stacked Histogram (Linear Scale) for {region}")
    plt.xlabel("Bin")
    plt.ylabel("Events")
    plt.legend()
    plt.show()

# Plot stacked histograms for each region in log scale
for region in regions:
    plt.figure(figsize=(10, 6))
    bin_centers = histograms[region][background_processes[0]][0]
    bin_width = bin_centers[1] - bin_centers[0]  # Assuming uniform bin width

    # Initialize an array to accumulate the stacked values
    cumulative_values = np.zeros_like(bin_centers)

    # Plot each background process stacked in log scale
    for i, process in enumerate(background_processes):
        if process in histograms[region]:
            y_values = histograms[region][process][1]
            plt.bar(bin_centers, y_values, width=bin_width, bottom=cumulative_values,
                    color=colors[i % len(colors)], alpha=0.9, label=process, align='center')
            cumulative_values += y_values

    # Plot the data points as black markers with error bars, no connecting line
    if "data" in histograms[region]:
        plt.errorbar(histograms[region]["data"][0], histograms[region]["data"][1], 
                     yerr=np.sqrt(histograms[region]["data"][1]), fmt='o', color='black', label="Data")

    # Set the y-axis limits based on the region
    if region == "CR1":
        plt.ylim(0.01, 20)
    elif region == "CR2":
        plt.ylim(0.01, 70)

    plt.yscale('log')  # Set y-axis to logarithmic scale

    plt.title(f"Stacked Histogram (Log Scale) for {region}")
    plt.xlabel("Bin")
    plt.ylabel("Events (Log Scale)")
    plt.legend()
    plt.show()
