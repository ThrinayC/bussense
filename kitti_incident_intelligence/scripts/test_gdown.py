try:
    import gdown
    print("gdown version:", gdown.__version__)
except ImportError:
    print("gdown not installed")
