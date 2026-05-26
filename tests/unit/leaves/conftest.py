"""matplotlib must use a headless backend in tests."""
import matplotlib

matplotlib.use("Agg")
