#!/usr/bin/env python
import os
import sys
import logging
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_kernels():
    """Get a list of available Jupyter kernels."""
    try:
        kernel_spec_manager = KernelSpecManager()
        kernel_specs = kernel_spec_manager.find_kernel_specs()
        logger.info(f"Found kernel specs: {kernel_specs}")
        return list(kernel_specs.keys())
    except Exception as e:
        logger.error(f"Error getting available kernels: {str(e)}")
        return []

def main():
    logger.info("Checking for available Jupyter kernels...")
    available_kernels = get_available_kernels()
    
    if available_kernels:
        logger.info(f"Found {len(available_kernels)} available kernels:")
        for i, kernel in enumerate(available_kernels, 1):
            logger.info(f"  {i}. {kernel}")
    else:
        logger.warning("No Jupyter kernels found!")
    
    # Check environment variables that might affect kernel discovery
    logger.info("\nEnvironment variables:")
    for var in ['JUPYTER_PATH', 'JUPYTER_CONFIG_DIR', 'JUPYTER_DATA_DIR', 'JUPYTER_RUNTIME_DIR', 'JUPYTER_PLATFORM_DIRS']:
        logger.info(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    # Check Python path
    logger.info("\nPython path:")
    for path in sys.path:
        logger.info(f"  {path}")

if __name__ == "__main__":
    main() 