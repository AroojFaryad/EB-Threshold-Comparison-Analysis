import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# ======================= CONFIGURATION =======================
class Config:
    BASE_PATH = 'D:/arooj/DATA'  # ← UPDATE THIS TO YOUR PATH
    OUTPUT_PATH = 'results'
    THRESHOLD_FACTOR = 1.5
    PIXEL_SCALE = 0.162  # arcsec/pixel
    RSUN = 695.7  # Solar radius in Mm
    RADIUS_ARCSEC = 960  # Solar angular radius
    MIN_LIFETIME = 1  # minutes
    MAX_LIFETIME = 10  # minutes
    BIN_WIDTH = 0.5
    
    # Dataset information
    DATASETS = [
        {"date": "20140614", "file": "sources_S2P0T10.5d3t5.npy", 
         "qs_mean": 152.28, "cadence": 11.4, "mu": 0.92},
        {"date": "20140615", "file": "sources_S2P0T14d3t5.npy", 
         "qs_mean": 108.35, "cadence": 11.4, "mu": 0.84},
        {"date": "20130906", "file": "sources_S2P0T14.5d3t10.npy", 
         "qs_mean": 150.84, "cadence": 5.5, "mu": 0.57},
        {"date": "20140906", "file": "sources_S2P0T14.5d3t5.npy", 
         "qs_mean": 148.91, "cadence": 11.6, "mu": 0.41},
        {"date": "20140905", "file": "sources_S2P0T15d3t5.npy", 
         "qs_mean": 111.70, "cadence": 11.6, "mu": 0.59},
        {"date": "20140909", "file": "sources_S2P0T6.2d3t5.npy", 
         "qs_mean": 168.10, "cadence": 11.6, "mu": 0.89},
        {"date": "20160903", "file": "sources_S2P0T7.5d3t3.npy", 
         "qs_mean": 179.41, "cadence": 20, "mu": 0.81},
        {"date": "20160904", "file": "sources_S2P0T11.5d3t3.npy", 
         "qs_mean": 172.28, "cadence": 20, "mu": 0.91},
    ]

# ======================= UTILITY FUNCTIONS =======================

def calculate_mm_conversion():
    """Calculate conversion factor from arcsec² to Mm²."""
    return Config.RSUN / Config.RADIUS_ARCSEC

def load_sources_data(dataset_info):
    """Load sources data from numpy file."""
    path = os.path.join(Config.BASE_PATH, dataset_info["date"], dataset_info["file"])
    print(f"Loading: {path}")
    
    if not os.path.exists(path):
        print(f" File not found: {path}")
        return None
    
    try:
        sources = np.load(path)
        print(f" Loaded {len(sources)} sources from {dataset_info['date']}")
        return sources
    except Exception as e:
        print(f" Error loading {path}: {str(e)}")
        return None

def calculate_lifetime(eb_data, cadence):
    """Calculate lifetime of EB event in minutes."""
    times = eb_data["timeframe"]
    return (times.max() - times.min() + 1) * cadence / 60

def convert_area(area_pixels, mu):
    """Convert area from pixels to arcsec² and Mm²."""
    area_arcsec = area_pixels * (Config.PIXEL_SCALE ** 2)
    mm_per_arcsec = calculate_mm_conversion()
    area_mm = area_arcsec * (mm_per_arcsec ** 2) / mu
    return area_arcsec, area_mm

# ======================= ANALYSIS FUNCTIONS =======================

def analyze_datasets():
    """Main analysis function for all datasets."""
    mm_per_arcsec = calculate_mm_conversion()
    
    # Initialize data containers
    results = {
        "lifetimes": {"all": [], "thresh": []},
        "areas": {"arcsec_all": [], "mm_all": [], "arcsec_thresh": [], "mm_thresh": []},
        "contrasts": {"all": [], "thresh": []}
    }
    
    successful_datasets = 0
    
    for dataset in Config.DATASETS:
        print(f"\n Processing dataset: {dataset['date']}")
        sources = load_sources_data(dataset)
        if sources is None:
            continue
            
        successful_datasets += 1
        threshold = Config.THRESHOLD_FACTOR * dataset["qs_mean"]
        print(f"Threshold: {threshold:.2f}")
        
        # Analyze each EB
        eb_ids = np.unique(sources["id"])
        print(f"Found {len(eb_ids)} EB events")
        
        for eb_id in eb_ids:
            eb = sources[sources["id"] == eb_id]
            lifetime = calculate_lifetime(eb, dataset["cadence"])
            
            if Config.MIN_LIFETIME <= lifetime < Config.MAX_LIFETIME:
                results["lifetimes"]["all"].append(lifetime)
                if np.any(eb["peak"] > threshold):
                    results["lifetimes"]["thresh"].append(lifetime)
        
        # Analyze areas
        area_arcsec, area_mm = convert_area(sources["area"], dataset["mu"])
        mask = area_arcsec > 0.01
        results["areas"]["arcsec_all"].extend(area_arcsec[mask])
        results["areas"]["mm_all"].extend(area_mm[mask])
        
        # Thresholded areas
        filtered = sources[sources["peak"] > threshold]
        area_arcsec_f, area_mm_f = convert_area(filtered["area"], dataset["mu"])
        mask_f = area_arcsec_f > 0.01
        results["areas"]["arcsec_thresh"].extend(area_arcsec_f[mask_f])
        results["areas"]["mm_thresh"].extend(area_mm_f[mask_f])
        
        # Analyze contrasts
        contrast_all = sources["peak"] / dataset["qs_mean"]
        results["contrasts"]["all"].extend(contrast_all[contrast_all > 1])
        
        contrast_thresh = contrast_all[(sources["peak"] > threshold) & (contrast_all > 1)]
        results["contrasts"]["thresh"].extend(contrast_thresh)
    
    print(f"\n Successfully processed {successful_datasets}/{len(Config.DATASETS)} datasets")
    
    # Convert to numpy arrays
    for category in results.values():
        for key in category:
            category[key] = np.array(category[key])
    
    return results

# ======================= PLOTTING FUNCTIONS =======================

def create_comparison_plot(results, output_path):
    """ Create the triple-panel comparison plot."""
    if len(results["lifetimes"]["all"]) == 0:
        print(" No data to plot! Check your file paths and data.")
        return
    
    plt.figure(figsize=(18, 6))
    
    # Lifetime plot
    plt.subplot(1, 3, 1)
    bins_lifetime = np.arange(0, Config.MAX_LIFETIME + Config.BIN_WIDTH, Config.BIN_WIDTH)
    plt.hist(results["lifetimes"]["all"], bins=bins_lifetime, alpha=0.5, 
             label="All Detections", density=True, color="gray", edgecolor="black")
    plt.hist(results["lifetimes"]["thresh"], bins=bins_lifetime, alpha=0.7, 
             label="Thresholded (Peak Threshold × QS)", density=True, 
             color="skyblue", edgecolor="black")
    plt.xlabel("Lifetime [min]")
    plt.ylabel("Fraction")
    plt.title("Lifetime Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Area plot
    plt.subplot(1, 3, 2)
    bins_area = np.arange(0, 3.1, 0.1)
    plt.hist(results["areas"]["arcsec_all"], bins=bins_area, alpha=0.5, 
             label="Detection (arcsec²)", density=True, color="gray", edgecolor="black")
    plt.hist(results["areas"]["mm_all"], bins=bins_area, alpha=0.5, 
             label="Detection (Mm²)", density=True, color="gray", 
             edgecolor="black", histtype="step", linestyle="--", linewidth=2)
    plt.hist(results["areas"]["arcsec_thresh"], bins=bins_area, alpha=0.5, 
             label="Thresholded (arcsec²)", density=True, 
             color="skyblue", edgecolor="black")
    plt.hist(results["areas"]["mm_thresh"], bins=bins_area, alpha=0.5, 
             label="Thresholded (Mm²)", density=True, 
             color="skyblue", edgecolor="blue", histtype="step", linestyle="--", linewidth=2)
    plt.xlabel("Area")
    plt.title("Area Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Contrast plot
    plt.subplot(1, 3, 3)
    bins_contrast = np.linspace(1, 2.5, 30)
    plt.hist(results["contrasts"]["all"], bins=bins_contrast, alpha=0.5, 
             label="All Detections", density=True, color="gray", edgecolor="black")
    plt.hist(results["contrasts"]["thresh"], bins=bins_contrast, alpha=0.7, 
             label="Thresholded", density=True, color="skyblue", edgecolor="black")
    plt.xlabel("Contrast [I/I$_{QS}$]")
    plt.title("Contrast Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Plot saved to: {output_path}")
    plt.show()

def print_statistics(results):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\n Lifetime:")
    if len(results['lifetimes']['all']) > 0:
        print(f"   All: {len(results['lifetimes']['all'])} events, "
              f"Median: {np.median(results['lifetimes']['all']):.2f} min")
        print(f"   Thresh: {len(results['lifetimes']['thresh'])} events, "
              f"Median: {np.median(results['lifetimes']['thresh']):.2f} min")
    else:
        print("   No lifetime data available")
    
    print(f"\n Area (arcsec²):")
    if len(results['areas']['arcsec_all']) > 0:
        print(f"   All: {len(results['areas']['arcsec_all'])}, "
              f"Median: {np.median(results['areas']['arcsec_all']):.3f}")
        print(f"   Thresh: {len(results['areas']['arcsec_thresh'])}, "
              f"Median: {np.median(results['areas']['arcsec_thresh']):.3f}")
    else:
        print("   No area data available")
    
    print(f"\n Contrast:")
    if len(results['contrasts']['all']) > 0:
        print(f"   All: {len(results['contrasts']['all'])}, "
              f"Median: {np.median(results['contrasts']['all']):.3f}")
        print(f"   Thresh: {len(results['contrasts']['thresh'])}, "
              f"Median: {np.median(results['contrasts']['thresh']):.3f}")
    else:
        print("No contrast data available")

# ======================= MAIN EXECUTION =======================

if __name__ == "__main__":
    # Create output directory
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    print(" Analyzing EB properties across datasets...")
    print(f"Data path: {Config.BASE_PATH}")
    print(f"Number of datasets: {len(Config.DATASETS)}")
    
    results = analyze_datasets()
    
    # Create and save plot
    output_file = os.path.join(Config.OUTPUT_PATH, "eb_property_comparison.pdf")
    create_comparison_plot(results, output_file)
    
    # Print statistics
    print_statistics(results)
    
    # Save numerical results
    if any(len(data) > 0 for category in results.values() for data in category.values()):
        np.savez(os.path.join(Config.OUTPUT_PATH, "analysis_results.npz"), **results)
        print(f"\n Numerical results saved to: {Config.OUTPUT_PATH}/analysis_results.npz")
    
    print(f"\n Analysis complete!")
